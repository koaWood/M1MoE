// ExpertCache.swift — DRAM-resident expert weight cache with LRU eviction.
//
// Implements the Apple paper's "windowing" strategy explicitly:
// recently-used experts stay in DRAM; cold experts are evicted to make room
// for the next layer's needs.
//
// Layout on disk: packed_experts/layer_NN.bin
//   [expert_0 bytes][expert_1 bytes]...[expert_N-1 bytes]
//   Each expert block = expertBytes bytes (gate+up+down, packed 4-bit)

import Darwin
import Foundation
import Metal

// MARK: - ExpertKey

private struct ExpertKey: Hashable {
    let layer: Int
    let index: Int
}

// MARK: - ExpertCache

public actor ExpertCache {

    private let spec:        ModelSpec
    private let hardware:    HardwareProfile
    private let expertBytes: Int

    // File descriptors, one per layer
    private let fds: [Int32]

    // DRAM cache: key → aligned buffer
    private var cache:    [ExpertKey: UnsafeMutableRawPointer] = [:]
    private var lruOrder: [ExpertKey] = []   // front = oldest

    // Budget
    private let maxCachedExperts: Int
    private let alignment = 2 * 1024 * 1024  // 2MB — matches Metal buffer alignment

    // MARK: - Init

    public init(spec: ModelSpec, hardware: HardwareProfile) throws {
        self.spec        = spec
        self.hardware    = hardware
        self.expertBytes = spec.expertBytes4Bit

        let modelDir = URL(fileURLWithPath: spec.modelDir ?? ".")
        let subdir   = modelDir.appendingPathComponent("packed_experts")

        var arr       = [Int32](repeating: -1, count: spec.numLayers)
        var available = 0
        for i in 0 ..< spec.numLayers {
            let path = subdir.appendingPathComponent(
                String(format: "layer_%02d.bin", i)).path
            let fd = Darwin.open(path, O_RDONLY)
            if fd >= 0 {
                _ = Darwin.fcntl(fd, F_RDAHEAD, 0)  // disable read-ahead; we do our own prefetch
                arr[i] = fd
                available += 1
            }
        }
        self.fds = arr

        self.maxCachedExperts = hardware.expertWindowCount(
            expertBytes: expertBytes,
            numLayers:   spec.numLayers,
            numExperts:  spec.moe.numExperts
        )

        print("[cache] \(spec.name): \(available)/\(spec.numLayers) layer files open")
        print("[cache] window: \(maxCachedExperts) experts " +
              "(\(maxCachedExperts * expertBytes / 1_048_576) MB of " +
              "\(hardware.availableForModelGB * 1024) MB available)")
    }

    deinit {
        for fd in fds where fd >= 0 { Darwin.close(fd) }
        for (_, ptr) in cache { Darwin.free(ptr) }
    }

    // MARK: - Fetch

    /// Return a pointer to expert `index` in `layer`, loading from SSD if needed.
    /// The pointer is valid until the next eviction (i.e., until the next fetch call
    /// that triggers eviction). Callers must finish using the pointer before calling
    /// fetch again on a full cache.
    public func fetch(layer: Int, index: Int) -> UnsafeRawPointer? {
        let key = ExpertKey(layer: layer, index: index)

        // Cache hit — promote to MRU
        if let ptr = cache[key] {
            promote(key)
            return UnsafeRawPointer(ptr)
        }

        // Evict if at capacity
        while cache.count >= maxCachedExperts, let evict = lruOrder.first {
            lruOrder.removeFirst()
            if let ptr = cache.removeValue(forKey: evict) {
                Darwin.free(ptr)
            }
        }

        // Allocate aligned buffer and pread from SSD
        guard layer < fds.count, fds[layer] >= 0 else { return nil }

        var rawPtr: UnsafeMutableRawPointer? = nil
        Darwin.posix_memalign(&rawPtr, alignment, expertBytes)
        guard let ptr = rawPtr else { return nil }

        let offset = off_t(index) * off_t(expertBytes)
        let n = Darwin.pread(fds[layer], ptr, expertBytes, offset)
        guard n == expertBytes else {
            Darwin.free(ptr)
            return nil
        }

        cache[key] = ptr
        lruOrder.append(key)
        return UnsafeRawPointer(ptr)
    }

    /// Prefetch a set of (layer, index) pairs concurrently using Swift TaskGroup.
    /// Returns a map of key → pointer for all successfully loaded experts.
    public func prefetch(experts: [(layer: Int, index: Int)]) async -> [(layer: Int, index: Int, ptr: UnsafeRawPointer?)] {
        // Collect what's already cached vs. what needs loading
        var results = [(layer: Int, index: Int, ptr: UnsafeRawPointer?)]()
        var toLoad  = [(layer: Int, index: Int)]()

        for e in experts {
            let key = ExpertKey(layer: e.layer, index: e.index)
            if let ptr = cache[key] {
                promote(key)
                results.append((e.layer, e.index, UnsafeRawPointer(ptr)))
            } else {
                toLoad.append(e)
            }
        }

        if toLoad.isEmpty { return results }

        // Evict to make room
        let needed = toLoad.count
        while cache.count + needed > maxCachedExperts, let evict = lruOrder.first {
            lruOrder.removeFirst()
            if let ptr = cache.removeValue(forKey: evict) {
                Darwin.free(ptr)
            }
        }

        // Allocate and load in parallel.
        // Raw pointers are captured as Int (bitPattern:) across task boundaries —
        // the same trick used in flash-moe to satisfy Swift's Sendable checker
        // without unsafely conforming UnsafeMutableRawPointer to Sendable.
        let expertBytesLocal = expertBytes
        let fdsLocal = fds
        let alignLocal = alignment

        // (layer, index, ptrInt): ptrInt == 0 means load failed
        typealias LoadResult = (layer: Int, index: Int, ptrInt: Int)
        let loaded: [LoadResult] = await withTaskGroup(of: LoadResult.self) { group in
            for e in toLoad {
                let layer = e.layer
                let index = e.index
                group.addTask {
                    guard layer < fdsLocal.count, fdsLocal[layer] >= 0 else {
                        return (layer, index, 0)
                    }
                    var rawPtr: UnsafeMutableRawPointer? = nil
                    Darwin.posix_memalign(&rawPtr, alignLocal, expertBytesLocal)
                    guard let ptr = rawPtr else { return (layer, index, 0) }
                    let offset = off_t(index) * off_t(expertBytesLocal)
                    let n = Darwin.pread(fdsLocal[layer], ptr, expertBytesLocal, offset)
                    if n != expertBytesLocal { Darwin.free(ptr); return (layer, index, 0) }
                    return (layer, index, Int(bitPattern: ptr))
                }
            }
            var out = [LoadResult]()
            for await r in group { out.append(r) }
            return out
        }

        for r in loaded {
            if r.ptrInt != 0, let ptr = UnsafeMutableRawPointer(bitPattern: r.ptrInt) {
                let key = ExpertKey(layer: r.layer, index: r.index)
                cache[key] = ptr
                lruOrder.append(key)
                results.append((r.layer, r.index, UnsafeRawPointer(ptr)))
            } else {
                results.append((r.layer, r.index, nil))
            }
        }

        return results
    }

    // MARK: - Warmup

    /// Touch the first 4KB of each layer file to warm OS file metadata.
    public func warmup() {
        var dummy = [UInt8](repeating: 0, count: 4096)
        for fd in fds where fd >= 0 {
            _ = Darwin.pread(fd, &dummy, dummy.count, 0)
        }
        print("[cache] warmup done (\(fds.filter { $0 >= 0 }.count) layers)")
    }

    // MARK: - Private

    private func promote(_ key: ExpertKey) {
        lruOrder.removeAll { $0 == key }
        lruOrder.append(key)
    }
}

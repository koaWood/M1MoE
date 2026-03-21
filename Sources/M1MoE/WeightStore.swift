// WeightStore.swift — Non-expert weight storage (mmap'd, read-only).
//
// Loads model_weights.bin + model_weights.json at startup.
// All non-expert tensors (embeddings, attention projections, norms, lm_head)
// live here for the lifetime of the process.
//
// Binary layout: model_weights.bin is a flat packed file.
// Manifest:      model_weights.json maps tensor name → {offset, shape, dtype}

import Foundation
import Metal

// MARK: - TensorInfo

public struct TensorInfo: Codable {
    public let offset: Int
    public let shape:  [Int]
    public let dtype:  String   // "bf16" | "f32" | "u32"

    public var numElements: Int { shape.reduce(1, *) }
    public var byteWidth: Int {
        switch dtype {
        case "bf16": return 2
        case "f32":  return 4
        case "u32":  return 4
        default:     return 2
        }
    }
    public var byteSize: Int { numElements * byteWidth }
}

// MARK: - WeightStore

public final class WeightStore: @unchecked Sendable {

    private let base:     UnsafeRawPointer
    private let fileSize: Int
    private let fd:       Int32
    private let manifest: [String: TensorInfo]

    // Exposed for Metal zero-copy wrapping
    public let mutableBase: UnsafeMutableRawPointer
    public var wfBuf: MTLBuffer? = nil   // set after MetalContext is ready

    // MARK: - Init

    public init(binURL: URL, jsonURL: URL) throws {
        // Load manifest
        let jsonData = try Data(contentsOf: jsonURL)
        self.manifest = try JSONDecoder().decode([String: TensorInfo].self, from: jsonData)

        // mmap the binary
        let fd = Darwin.open(binURL.path, O_RDONLY)
        guard fd >= 0 else {
            throw WeightStoreError.fileNotFound(binURL.path)
        }
        var st = stat()
        Darwin.fstat(fd, &st)
        let size = Int(st.st_size)

        let ptr = Darwin.mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0)
        guard ptr != MAP_FAILED, let rawPtr = ptr else {
            Darwin.close(fd)
            throw WeightStoreError.mmapFailed
        }

        self.fd       = fd
        self.fileSize = size
        self.base     = UnsafeRawPointer(rawPtr)
        self.mutableBase = UnsafeMutableRawPointer(rawPtr)

        let mb = Double(size) / 1_048_576
        print(String(format: "[weights] mmap'd %.2f GB from %@",
                     mb / 1024, binURL.lastPathComponent))
        print("[weights] \(manifest.count) tensors loaded from manifest")
    }

    deinit {
        Darwin.munmap(mutableBase, fileSize)
        Darwin.close(fd)
    }

    // MARK: - Accessors

    public func info(named name: String) -> TensorInfo? {
        manifest[name]
    }

    /// Raw pointer to tensor data (BF16 or packed u32).
    public func pointer(named name: String) -> UnsafeRawPointer? {
        guard let info = manifest[name] else { return nil }
        return base.advanced(by: info.offset)
    }

    /// Typed pointer to BF16 weight tensor.
    public func bf16(named name: String) -> UnsafePointer<UInt16>? {
        pointer(named: name)?.bindMemory(to: UInt16.self, capacity: 1)
    }

    /// Typed pointer to packed u32 weight tensor.
    public func u32(named name: String) -> UnsafePointer<UInt32>? {
        pointer(named: name)?.bindMemory(to: UInt32.self, capacity: 1)
    }

    /// Typed pointer to f32 tensor.
    public func f32(named name: String) -> UnsafePointer<Float>? {
        pointer(named: name)?.bindMemory(to: Float.self, capacity: 1)
    }

    /// Offset (bytes) of a tensor from the start of the weight file.
    /// Used to encode Metal buffer offsets without copying data.
    public func offset(named name: String) -> Int? {
        manifest[name]?.offset
    }

    // MARK: - Metal zero-copy

    /// Wrap the entire mmap region as a shared MTLBuffer (zero copy).
    public func makeMetalBuffer(device: MTLDevice) -> MTLBuffer? {
        let buf = device.makeBuffer(bytesNoCopy: mutableBase,
                                    length: fileSize,
                                    options: .storageModeShared,
                                    deallocator: nil)
        wfBuf = buf
        return buf
    }

    // MARK: - Diagnostics

    public func has(_ name: String) -> Bool { manifest[name] != nil }
}

// MARK: - Errors

public enum WeightStoreError: Error {
    case fileNotFound(String)
    case mmapFailed
}

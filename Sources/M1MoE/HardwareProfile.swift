// HardwareProfile.swift — Detected hardware capabilities.
//
// Drives buffer sizing, kernel selection, and expert window decisions.
// Detected once at startup; all engine components read from this.

import Foundation
import Metal

// MARK: - HardwareProfile

public struct HardwareProfile: Sendable {

    /// GPU family — drives which kernel variants to use.
    public enum GPUTier: Sendable {
        case apple3          // M1, M2 — 8-core GPU
        case apple6          // M2 Pro/Max, M3 — 30-40 core GPU
        case apple9          // M3 Max, M4 family
        case unknown
    }

    public let deviceName:       String
    public let gpuTier:          GPUTier
    public let unifiedMemoryGB:  Int     // total system RAM
    public let availableForModelGB: Int  // conservative estimate for model use

    // ── Derived limits ──────────────────────────────────────────────────────

    /// DRAM budget (bytes) to use for expert window cache.
    /// Leaves headroom for OS, KV cache, and non-expert weights.
    public var expertWindowBytes: Int {
        // Use 60% of available memory for expert window cache
        availableForModelGB * 1_073_741_824 * 60 / 100
    }

    /// Optimal threadgroup width for this GPU tier.
    public var threadgroupWidth: Int {
        switch gpuTier {
        case .apple3:  return 32
        case .apple6:  return 64
        case .apple9:  return 64
        case .unknown: return 32
        }
    }

    // ── Detection ───────────────────────────────────────────────────────────

    public static func detect() -> HardwareProfile {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return HardwareProfile(deviceName: "Unknown", gpuTier: .unknown,
                                   unifiedMemoryGB: 8, availableForModelGB: 4)
        }

        let name       = device.name
        let totalBytes = ProcessInfo.processInfo.physicalMemory
        let totalGB    = Int(totalBytes / 1_073_741_824)

        // Leave ~4GB for OS + app overhead
        let availableGB = max(totalGB - 4, 1)

        let tier: GPUTier
        if name.contains("M1") || name.contains("M2") {
            tier = .apple3
        } else if name.contains("M3") {
            tier = .apple6
        } else if name.contains("M4") {
            tier = .apple9
        } else {
            tier = .unknown
        }

        return HardwareProfile(deviceName: name, gpuTier: tier,
                               unifiedMemoryGB: totalGB,
                               availableForModelGB: availableGB)
    }

    // ── Expert window size ───────────────────────────────────────────────────

    /// Compute how many experts to keep cached in the window,
    /// given the per-expert byte size from a ModelSpec.
    public func expertWindowCount(expertBytes: Int, numLayers: Int, numExperts: Int) -> Int {
        guard expertBytes > 0 else { return 0 }
        let maxExperts = expertWindowBytes / expertBytes
        let totalExperts = numLayers * numExperts
        return min(maxExperts, totalExperts)
    }
}

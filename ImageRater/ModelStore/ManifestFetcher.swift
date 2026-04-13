import CryptoKit
import Foundation

struct ModelManifest: Decodable {
    struct ModelEntry: Decodable {
        let name: String
        let version: String
        let url: URL
        let sha256: String
    }
    let models: [ModelEntry]
    let signature: String
}

enum ManifestFetcher {
    // Hardcoded manifest URL — not user-configurable. Replace before shipping.
    private static let manifestURL: URL = {
        #if DEBUG
        if let override = ProcessInfo.processInfo.environment["IMAGERATING_MANIFEST_URL"],
           let url = URL(string: override) { return url }
        #endif
        return URL(string: "https://REPLACE_WITH_REAL_MANIFEST_HOST/models-manifest.json")!
    }()

    // Ed25519 public key hex. Replace before shipping. Override in DEBUG via env var.
    private static let publicKeyHex: String = {
        #if DEBUG
        if let override = ProcessInfo.processInfo.environment["IMAGERATING_PUBKEY_HEX"] {
            return override
        }
        #endif
        return "REPLACE_WITH_REAL_ED25519_PUBLIC_KEY_HEX"
    }()

    static func fetch() async throws -> ModelManifest {
        let (data, _) = try await URLSession.shared.data(from: manifestURL)
        let manifest = try JSONDecoder().decode(ModelManifest.self, from: data)
        try verifySignature(of: data, manifest: manifest)
        return manifest
    }

    private static func verifySignature(of data: Data, manifest: ModelManifest) throws {
        guard publicKeyHex != "REPLACE_WITH_REAL_ED25519_PUBLIC_KEY_HEX",
              let pubKeyData = Data(hexString: publicKeyHex),
              let sigData = Data(hexString: manifest.signature) else {
            // In development (placeholder key), skip signature verification
            #if DEBUG
            return
            #else
            throw ModelStoreError.manifestVerificationFailed
            #endif
        }
        let pubKey = try Curve25519.Signing.PublicKey(rawRepresentation: pubKeyData)
        // Strip signature field before verifying canonical JSON
        guard var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelStoreError.manifestVerificationFailed
        }
        json.removeValue(forKey: "signature")
        let canonical = try JSONSerialization.data(withJSONObject: json, options: .sortedKeys)
        guard pubKey.isValidSignature(sigData, for: canonical) else {
            throw ModelStoreError.manifestVerificationFailed
        }
    }
}

private extension Data {
    init?(hexString: String) {
        guard hexString.count % 2 == 0 else { return nil }
        var data = Data(capacity: hexString.count / 2)
        var index = hexString.startIndex
        while index < hexString.endIndex {
            let next = hexString.index(index, offsetBy: 2)
            guard let byte = UInt8(hexString[index..<next], radix: 16) else { return nil }
            data.append(byte)
            index = next
        }
        self = data
    }
}

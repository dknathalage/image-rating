// Interface targeting LibRaw 0.21+
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

/// Camera/lens/shooting metadata extracted directly from LibRaw's parsed makernotes.
/// Fields are nil/0 when not available in the file.
@interface LibRawMetadata : NSObject
@property (nonatomic, copy, nullable)   NSString *make;
@property (nonatomic, copy, nullable)   NSString *model;
@property (nonatomic, copy, nullable)   NSString *normalizedMake;
@property (nonatomic, copy, nullable)   NSString *normalizedModel;
@property (nonatomic, copy, nullable)   NSString *lens;
@property (nonatomic, copy, nullable)   NSString *software;
@property (nonatomic, assign)           float     iso;           // 0 = unknown
@property (nonatomic, assign)           float     shutterSpeed;  // seconds; 0 = unknown
@property (nonatomic, assign)           float     aperture;      // f-number; 0 = unknown
@property (nonatomic, assign)           float     focalLength;   // mm; 0 = unknown
@property (nonatomic, assign)           float     focalLength35mm; // 0 = unknown
@property (nonatomic, assign)           int       pixelWidth;
@property (nonatomic, assign)           int       pixelHeight;
@property (nonatomic, assign)           int       meteringMode;  // EXIF metering mode value
@property (nonatomic, assign)           int       exposureProgram; // EXIF exposure program value
@property (nonatomic, copy, nullable)   NSDate   *dateTaken;
@end

@interface LibRawBridge : NSObject
/// Extracts embedded JPEG preview only. Returns nil if no embedded preview exists.
/// Use for thumbnails — never triggers full RAW decode.
+ (nullable CGImageRef)previewAtPath:(NSString *)path CF_RETURNS_RETAINED;
/// Full decode: embedded JPEG preview first, falls back to full LibRaw decode.
/// Use for processing pipeline only, not thumbnails.
+ (nullable CGImageRef)decodeFileAtPath:(NSString *)path CF_RETURNS_RETAINED;
/// Extract camera/lens/shooting metadata via LibRaw. Returns nil if file cannot be opened.
/// Use for RAW files; non-RAW should fall back to CGImageSource.
+ (nullable LibRawMetadata *)metadataAtPath:(NSString *)path;
@end

#import "LibRawBridge.h"
#import <libraw/libraw.h>

@implementation LibRawMetadata
@end

static NSString *_librawStr(const char *buf) {
    if (!buf || buf[0] == '\0') return nil;
    NSString *s = [NSString stringWithUTF8String:buf];
    s = [s stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
    return s.length > 0 ? s : nil;
}

@implementation LibRawBridge

+ (CGImageRef)previewAtPath:(NSString *)path {
    // Heap-allocate: LibRaw struct is ~600 KB — stack allocation overflows secondary thread stacks.
    LibRaw *processor = new LibRaw();
    int ret = processor->open_file([path UTF8String]);
    if (ret != LIBRAW_SUCCESS) { delete processor; return nil; }

    CGImageRef img = nil;
    if (processor->unpack_thumb() == LIBRAW_SUCCESS) {
        libraw_processed_image_t *thumb = processor->dcraw_make_mem_thumb();
        if (thumb && thumb->type == LIBRAW_IMAGE_JPEG) {
            NSData *jpegData = [NSData dataWithBytes:thumb->data length:thumb->data_size];
            LibRaw::dcraw_clear_mem(thumb);
            CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)jpegData);
            img = CGImageCreateWithJPEGDataProvider(provider, nil, true, kCGRenderingIntentDefault);
            CGDataProviderRelease(provider);
        } else if (thumb) {
            LibRaw::dcraw_clear_mem(thumb);
        }
    }

    processor->recycle();
    delete processor;
    return img; // nil if no embedded JPEG — caller decides what to do
}

+ (CGImageRef)decodeFileAtPath:(NSString *)path {
    // Fast path: embedded JPEG preview
    CGImageRef preview = [self previewAtPath:path];
    if (preview) return preview;

    // Full decode fallback (for processing pipeline, not thumbnails)
    LibRaw *processor = new LibRaw();
    int ret = processor->open_file([path UTF8String]);
    if (ret != LIBRAW_SUCCESS) { delete processor; return nil; }
    if (processor->unpack() != LIBRAW_SUCCESS) { delete processor; return nil; }
    if (processor->dcraw_process() != LIBRAW_SUCCESS) { delete processor; return nil; }

    libraw_processed_image_t *image = processor->dcraw_make_mem_image();
    if (!image) { delete processor; return nil; }

    int width = image->width, height = image->height;
    int channels = image->colors;
    NSData *data = [NSData dataWithBytes:image->data length:image->data_size];
    LibRaw::dcraw_clear_mem(image);
    processor->recycle();
    delete processor;

    size_t bitsPerPixel = (size_t)channels * 8;
    size_t bytesPerRow = (size_t)width * channels;
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGBitmapInfo bitmapInfo = (channels == 4)
        ? (CGBitmapInfo)(kCGBitmapByteOrderDefault | kCGImageAlphaNoneSkipLast)
        : (CGBitmapInfo)(kCGBitmapByteOrderDefault | kCGImageAlphaNone);
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef img = CGImageCreate(width, height, 8, bitsPerPixel, bytesPerRow, space,
                                   bitmapInfo, provider, nil, false, kCGRenderingIntentDefault);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(space);
    return img;
}

+ (LibRawMetadata *)metadataAtPath:(NSString *)path {
    LibRaw *processor = new LibRaw();
    if (processor->open_file([path UTF8String]) != LIBRAW_SUCCESS) {
        delete processor;
        return nil;
    }

    LibRawMetadata *meta = [[LibRawMetadata alloc] init];
    libraw_data_t &d = processor->imgdata;

    // Camera identity
    meta.make             = _librawStr(d.idata.make);
    meta.model            = _librawStr(d.idata.model);
    meta.normalizedMake   = _librawStr(d.idata.normalized_make);
    meta.normalizedModel  = _librawStr(d.idata.normalized_model);
    meta.software         = _librawStr(d.idata.software);

    // Lens (Lens field in libraw_lensinfo_t; fall back to LensMake if blank)
    NSString *lens = _librawStr(d.lens.Lens);
    if (!lens) lens = _librawStr(d.lens.LensMake);
    meta.lens = lens;

    // Shooting parameters
    meta.iso             = d.other.iso_speed;
    meta.shutterSpeed    = d.other.shutter;
    meta.aperture        = d.other.aperture;
    meta.focalLength     = d.other.focal_len;
    meta.focalLength35mm = (float)d.lens.FocalLengthIn35mmFormat;

    // Output pixel dimensions after any flip/crop
    meta.pixelWidth  = d.sizes.iwidth;
    meta.pixelHeight = d.sizes.iheight;

    // Shooting info (metering, exposure program)
    meta.meteringMode    = d.shootinginfo.MeteringMode;
    meta.exposureProgram = d.shootinginfo.ExposureProgram;

    // Timestamp
    if (d.other.timestamp > 0) {
        meta.dateTaken = [NSDate dateWithTimeIntervalSince1970:(NSTimeInterval)d.other.timestamp];
    }

    processor->recycle();
    delete processor;
    return meta;
}

@end

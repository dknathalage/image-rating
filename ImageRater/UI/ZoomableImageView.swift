import AppKit
import SwiftUI

/// NSScrollView-backed zoomable image view.
/// Trackpad pinch zooms towards the cursor position natively.
struct ZoomableImageView: NSViewRepresentable {
    let image: NSImage?
    @Binding var scale: CGFloat
    let fitTrigger: Int

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> NSScrollView {
        let c = context.coordinator
        c.setup()
        return c.scrollView
    }

    func updateNSView(_ sv: NSScrollView, context: Context) {
        let c = context.coordinator
        c.onScaleChange = { s in scale = s }

        if c.loadedImage !== image {
            c.loadedImage = image
            c.imageView.image = image
            if let image {
                c.imageView.frame = CGRect(origin: .zero, size: image.size)
            }
            c.fitToView()
            return
        }

        if c.lastFitTrigger != fitTrigger {
            c.lastFitTrigger = fitTrigger
            c.fitToView()
            return
        }

        guard !c.internalUpdate, scale > 0,
              abs(sv.magnification - scale) > 0.005 else { return }
        c.internalUpdate = true
        sv.setMagnification(scale, centeredAt: NSPoint(
            x: sv.documentVisibleRect.midX,
            y: sv.documentVisibleRect.midY
        ))
        c.internalUpdate = false
    }

    // MARK: - Coordinator

    final class Coordinator: NSObject {
        let scrollView = NSScrollView()
        let imageView = NSImageView()
        var loadedImage: NSImage?
        var onScaleChange: ((CGFloat) -> Void)?
        var internalUpdate = false
        var lastFitTrigger = 0

        func setup() {
            // Replace default clip view with centering variant
            let clip = CenteringClipView()
            clip.drawsBackground = false
            scrollView.contentView = clip

            imageView.imageScaling = .scaleNone
            scrollView.documentView = imageView
            scrollView.allowsMagnification = true
            scrollView.minMagnification = 0.05
            scrollView.maxMagnification = 8.0
            scrollView.hasHorizontalScroller = true
            scrollView.hasVerticalScroller = true
            scrollView.autohidesScrollers = true
            scrollView.backgroundColor = NSColor.windowBackgroundColor
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(magnifyEnded),
                name: NSScrollView.didEndLiveMagnifyNotification,
                object: scrollView
            )
        }

        func fitToView(attempt: Int = 0) {
            guard let img = loadedImage else { return }
            let svSize = scrollView.contentSize
            guard svSize.width > 0, svSize.height > 0,
                  img.size.width > 0, img.size.height > 0 else {
                guard attempt < 5 else { return }
                DispatchQueue.main.async { self.fitToView(attempt: attempt + 1) }
                return
            }
            let fit = min(svSize.width / img.size.width, svSize.height / img.size.height)
            internalUpdate = true
            scrollView.magnification = fit
            // Force clip view to re-evaluate centering constraint
            scrollView.contentView.scroll(to: scrollView.contentView.bounds.origin)
            internalUpdate = false
            onScaleChange?(fit)
        }

        @objc func magnifyEnded(_ note: Notification) {
            guard !internalUpdate else { return }
            onScaleChange?(scrollView.magnification)
        }
    }
}

// MARK: - CenteringClipView

/// Keeps the document view centered when it is smaller than the visible area.
private final class CenteringClipView: NSClipView {
    override func constrainBoundsRect(_ proposedBounds: NSRect) -> NSRect {
        var rect = super.constrainBoundsRect(proposedBounds)
        guard let docFrame = documentView?.frame else { return rect }
        if rect.width > docFrame.width {
            rect.origin.x = (docFrame.width - rect.width) / 2
        }
        if rect.height > docFrame.height {
            rect.origin.y = (docFrame.height - rect.height) / 2
        }
        return rect
    }
}

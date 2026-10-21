use crate::{
    Counter, Format, Image,
    backend::{InnerSwapchain, Surface},
};

#[derive(Clone, Copy)]
pub struct SwapchainDescription {
    pub format: Format,
    pub frames_in_flight: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy)]
pub struct AcquiredImage {
    pub(crate) image: Image,
    pub(crate) image_index: usize,
    pub(crate) slot: usize,
    pub(crate) slot_count: usize,
    pub(crate) acquire_semaphore: ash::vk::Semaphore,
    pub(crate) present_semaphore: ash::vk::Semaphore,
}

impl AcquiredImage {
    #[inline]
    pub fn image(&self) -> Image { self.image }

    #[inline]
    pub fn frame_index(&self) -> usize { self.slot }

    #[inline]
    pub fn frames_in_flight(&self) -> usize { self.slot_count }
}

pub struct Swapchain {
    pub(crate) inner: InnerSwapchain,
    pub(crate) surface: Surface,
}

impl Swapchain {
    pub fn resize(&mut self, width: u32, height: u32) {
        crate::wait_idle();

        let swapchain_description = SwapchainDescription {
            format: self.inner.swapchain_description.format,
            frames_in_flight: self.inner.swapchain_description.frames_in_flight,
            width,
            height,
        };

        let new = InnerSwapchain::new(&self.surface, &swapchain_description, Some(&self.inner), self.inner.ctx);

        self.inner.cleanup();
        self.inner = new;
    }

    pub fn extent(&self) -> (u32, u32) { (self.inner.extent.width, self.inner.extent.height) }

    pub fn acquire_image(&mut self) -> AcquiredImage { self.inner.acquire_image() }

    pub fn frame_count(&self) -> usize { self.inner.actual_image_count }

    pub fn present(&mut self, accquired_image: &AcquiredImage, counter: Counter) { self.inner.present(accquired_image, counter); }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        crate::wait_idle();
        self.inner.cleanup();
        self.surface.cleanup();
    }
}

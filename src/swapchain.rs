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
    pub(crate) acquire_semaphore: ash::vk::Semaphore,
    pub(crate) present_semaphore: ash::vk::Semaphore,
}

impl AcquiredImage {
    #[inline]
    pub fn image(&self) -> Image {
        return self.image;
    }
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

        let new = InnerSwapchain::new(&self.surface, &swapchain_description, Some(&self.inner));

        self.inner.cleanup();
        self.inner = new;
    }

    pub fn acquire_image(&mut self) -> AcquiredImage {
        return self.inner.acquire_image();
    }

    pub fn present(&mut self, accquired_image: &AcquiredImage, counter: Counter) {
        self.inner.present(accquired_image, counter);
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        crate::wait_idle();
        self.inner.cleanup();
        self.surface.cleanup();
    }
}

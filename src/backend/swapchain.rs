use ash::vk::{self, FenceCreateFlags};
use gpu_allocator::vulkan::Allocation;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    Counter, Image, ImageView, ImageViewDescription,
    backend::{instance::Instance, resources::InnerImage},
    swapchain::{AcquiredImage, SwapchainDescription},
};

pub(crate) struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub(crate) struct Surface {
    pub(crate) handle: vk::SurfaceKHR,
    pub(crate) loader: ash::khr::surface::Instance,
}

impl Surface {
    pub(crate) fn get_swapchain_support(&self, physical_device: ash::vk::PhysicalDevice) -> Option<SwapchainSupport> {
        unsafe {
            let capabilities = self.loader.get_physical_device_surface_capabilities(physical_device, self.handle).ok()?;

            let formats = self.loader.get_physical_device_surface_formats(physical_device, self.handle).ok()?;

            let present_modes = self.loader.get_physical_device_surface_present_modes(physical_device, self.handle).ok()?;

            if formats.is_empty() || present_modes.is_empty() {
                return None;
            } else {
                return Some(SwapchainSupport {
                    capabilities,
                    formats,
                    present_modes,
                });
            }
        }
    }

    pub(crate) fn create_surface<W: HasDisplayHandle + HasWindowHandle>(instance: &Instance, window: &W) -> Surface {
        let raw_window = window.window_handle().unwrap().as_raw();
        let raw_display = window.display_handle().unwrap().as_raw();

        use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

        let surface_handle = match (raw_window, raw_display) {
            (RawWindowHandle::Win32(w), RawDisplayHandle::Windows(_)) => {
                let info = ash::vk::Win32SurfaceCreateInfoKHR::default().hinstance(w.hinstance.unwrap().get()).hwnd(w.hwnd.get());
                let loader = ash::khr::win32_surface::Instance::new(&instance.entry, &instance.handle);
                unsafe { loader.create_win32_surface(&info, None).expect("Failed to create surface") }
            }
            (RawWindowHandle::Xcb(w), RawDisplayHandle::Xcb(d)) => {
                let info = ash::vk::XcbSurfaceCreateInfoKHR::default().connection(d.connection.unwrap().as_ptr()).window(w.window.get());
                let loader = ash::khr::xcb_surface::Instance::new(&instance.entry, &instance.handle);
                unsafe { loader.create_xcb_surface(&info, None).expect("Failed to create surface") }
            }
            (RawWindowHandle::Xlib(w), RawDisplayHandle::Xlib(d)) => {
                let info = ash::vk::XlibSurfaceCreateInfoKHR::default().dpy(d.display.unwrap().as_ptr() as *mut _).window(w.window);
                let loader = ash::khr::xlib_surface::Instance::new(&instance.entry, &instance.handle);
                unsafe { loader.create_xlib_surface(&info, None).expect("Failed to create surface") }
            }
            (RawWindowHandle::Wayland(w), RawDisplayHandle::Wayland(d)) => {
                let info = ash::vk::WaylandSurfaceCreateInfoKHR::default().display(d.display.as_ptr()).surface(w.surface.as_ptr());
                let loader = ash::khr::wayland_surface::Instance::new(&instance.entry, &instance.handle);
                unsafe { loader.create_wayland_surface(&info, None).expect("Failed to create surface") }
            }
            (RawWindowHandle::AppKit(w), RawDisplayHandle::AppKit(_)) => {
                let info = ash::vk::MetalSurfaceCreateInfoEXT::default().layer(w.ns_view.as_ptr());
                let loader = ash::ext::metal_surface::Instance::new(&instance.entry, &instance.handle);
                unsafe { loader.create_metal_surface(&info, None).expect("Failed to create surface") }
            }

            _ => panic!("Unsupported platform or mismatched window/display handle"),
        };

        return Surface {
            handle: surface_handle,
            loader: ash::khr::surface::Instance::new(&instance.entry, &instance.handle),
        };
    }

    pub(crate) fn cleanup(&self) {
        unsafe {
            self.loader.destroy_surface(self.handle, None);
        }
    }
}

pub(crate) struct InnerSwapchain {
    handle: vk::SwapchainKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    pub(crate) swapchain_description: SwapchainDescription,

    images: Vec<Image>,
    acquire_semaphores: Vec<vk::Semaphore>,
    present_semaphores: Vec<vk::Semaphore>,
    fences: Vec<vk::Fence>,

    actual_image_count: usize,
    frame_timeline: usize,
}

impl InnerSwapchain {
    pub(crate) fn new(surface: &Surface, swapchain_description: &SwapchainDescription, old_swapchain: Option<&InnerSwapchain>) -> InnerSwapchain {
        let ctx = crate::CONTEXT.get().unwrap();

        let swapchain_loader = ash::khr::swapchain::Device::new(&ctx.instance.handle, &ctx.device.handle);

        let support = surface.get_swapchain_support(ctx.device.physical_device.handle).expect("Swapchain not supported!!");

        let present_mode = {
            if support.present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else {
                vk::PresentModeKHR::FIFO
            }
        };

        let surface_format = {
            support
                .formats
                .iter()
                .cloned()
                .find(|f| f.format == swapchain_description.format.to_vk() && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .unwrap_or_else(|| {
                    println!("Using default format for swapchain: {:?}", support.formats[0]);
                    support.formats[0]
                })
        };

        let extent = {
            if support.capabilities.current_extent.width != u32::MAX {
                support.capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: swapchain_description
                        .width
                        .clamp(support.capabilities.min_image_extent.width, support.capabilities.max_image_extent.width),
                    height: swapchain_description
                        .height
                        .clamp(support.capabilities.min_image_extent.height, support.capabilities.max_image_extent.height),
                }
            }
        };

        // Clamp requested image count to what the driver actually supports.
        // max_image_count == 0 means no upper limit.
        let image_count = {
            let min = support.capabilities.min_image_count;
            let max = support.capabilities.max_image_count;
            let requested = swapchain_description.frames_in_flight;
            if max == 0 { requested.max(min) } else { requested.clamp(min, max) }
        };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(match old_swapchain {
                Some(s) => s.handle,
                None => vk::SwapchainKHR::null(),
            });

        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None).expect("Failed to create swapchain") };

        // Query actual image count — driver may have created more than requested.
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain).expect("Failed to get swapchain images") };
        let actual_image_count = images.len();

        let inner_images: Vec<_> = images
            .iter()
            .map(|img| InnerImage {
                image: *img,
                mem_requirements: vk::MemoryRequirements::default(),
                format: swapchain_description.format.to_vk(),
                allocation: Allocation::default(),
            })
            .collect();

        let ids = inner_images
            .into_iter()
            .map(|img| {
                let inner_view = ctx.device.create_image_view(&img, &ImageViewDescription::default());
                let raw = inner_view.view;
                let view_id = ctx.image_views.write().unwrap().insert(inner_view);
                let image_id = ctx.images.write().unwrap().insert(img);

                Image {
                    default_view: ImageView {
                        raw,
                        id: view_id,
                    },
                    id: image_id,
                }
            })
            .collect();

        // All sync objects are sized to actual_image_count so no slot ever goes OOB,
        // regardless of what the driver handed back vs what the user requested.
        let acquire_semaphores = unsafe {
            (0..actual_image_count)
                .map(|_| ctx.device.handle.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap())
                .collect()
        };
        let present_semaphores = unsafe {
            (0..actual_image_count)
                .map(|_| ctx.device.handle.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap())
                .collect()
        };
        // Start fences pre-signaled so the first wait in acquire_image passes immediately.
        let fences = unsafe {
            (0..actual_image_count)
                .map(|_| ctx.device.handle.create_fence(&vk::FenceCreateInfo::default().flags(FenceCreateFlags::SIGNALED), None).unwrap())
                .collect()
        };

        return InnerSwapchain {
            handle: swapchain,
            swapchain_loader,
            images: ids,
            swapchain_description: *swapchain_description,
            acquire_semaphores,
            present_semaphores,
            fences,
            actual_image_count,
            frame_timeline: 0,
        };
    }

    pub(crate) fn acquire_image(&mut self) -> AcquiredImage {
        let ctx = crate::CONTEXT.get().unwrap();
        let slot = self.frame_timeline;

        // Block until the previous frame using this slot has finished on the GPU,
        // then reset the fence so present() can signal it again.
        unsafe {
            ctx.device.handle.wait_for_fences(&[self.fences[slot]], true, u64::MAX).unwrap();
            ctx.device.handle.reset_fences(&[self.fences[slot]]).unwrap();
        }

        let acquire_info = vk::AcquireNextImageInfoKHR::default()
            .swapchain(self.handle)
            .timeout(u64::MAX)
            .semaphore(self.acquire_semaphores[slot])
            .device_mask(1);

        let (index, _) = unsafe { self.swapchain_loader.acquire_next_image2(&acquire_info).expect("Failed to acquire image") };

        // Don't advance frame_timeline here — do it after present so the slot
        // isn't reused before the fence is handed off to the queue.
        return AcquiredImage {
            image: self.images[index as usize],
            image_index: index as usize,
            acquire_semaphore: self.acquire_semaphores[slot],
            present_semaphore: self.present_semaphores[slot],
            slot,
        };
    }

    pub(crate) fn present(&mut self, acquired_image: &AcquiredImage, counter: Counter) {
        let ctx = crate::CONTEXT.get().unwrap();
        let (queue_id, value) = counter.decode();
        let queue = &ctx.queues[queue_id as usize];

        unsafe {
            ctx.device
                .handle
                .queue_submit2(
                    queue.queue,
                    &[
                        vk::SubmitInfo2::default()
                            .wait_semaphore_infos(&[
                                vk::SemaphoreSubmitInfo {
                                    semaphore: queue.semaphore,
                                    value,
                                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                                    device_index: 0,
                                    ..Default::default()
                                },
                            ])
                            .signal_semaphore_infos(&[
                                vk::SemaphoreSubmitInfo {
                                    semaphore: acquired_image.present_semaphore,
                                    value: 0,
                                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                                    device_index: 0,
                                    ..Default::default()
                                },
                            ]),
                    ],
                    self.fences[acquired_image.slot],
                )
                .unwrap();
        }

        unsafe {
            self.swapchain_loader
                .queue_present(
                    queue.queue,
                    &vk::PresentInfoKHR::default()
                        .swapchains(&[self.handle])
                        .image_indices(&[acquired_image.image_index as u32])
                        .wait_semaphores(&[acquired_image.present_semaphore]),
                )
                .expect("Failed to present image");
        }

        // Advance only after present so the slot isn't reused until the fence is in flight.
        self.frame_timeline = (acquired_image.slot + 1) % self.actual_image_count;
    }

    pub(crate) fn cleanup(&self) {
        let ctx = crate::CONTEXT.get().unwrap();

        self.images.iter().for_each(|img| {
            ctx.images.write().unwrap().remove(img.id).unwrap();
            let view = ctx.image_views.write().unwrap().remove(img.default_view.id).unwrap();
            ctx.device.destroy_image_view(view);
        });

        unsafe {
            self.acquire_semaphores.iter().for_each(|s| ctx.device.handle.destroy_semaphore(*s, None));
            self.present_semaphores.iter().for_each(|s| ctx.device.handle.destroy_semaphore(*s, None));
            self.fences.iter().for_each(|f| ctx.device.handle.destroy_fence(*f, None));
            self.swapchain_loader.destroy_swapchain(self.handle, None);
        }
    }
}

/*pub(crate) struct InnerSwapchain {
    handle: vk::SwapchainKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    pub(crate) swapchain_description: SwapchainDescription,

    images: Vec<Image>,
    acquire_semaphores: Vec<vk::Semaphore>,
    present_semaphores: Vec<vk::Semaphore>,
    fence: Vec<vk::Fence>,
    slot_used: Vec<bool>,

    frame_timeline: usize,
}

impl InnerSwapchain {
    pub(crate) fn new(surface: &Surface, swapchain_description: &SwapchainDescription, old_swapchain: Option<&InnerSwapchain>) -> InnerSwapchain {
        let ctx = crate::CONTEXT.get().unwrap();

        let swapchain_loader = ash::khr::swapchain::Device::new(&ctx.instance.handle, &ctx.device.handle);

        let support = surface.get_swapchain_support(ctx.device.physical_device.handle).expect("Swapchain not supported!!");

        let present_mode = {
            if support.present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else {
                vk::PresentModeKHR::FIFO
            }
        };

        let surface_format = {
            support
                .formats
                .iter()
                .cloned()
                .find(|f| f.format == swapchain_description.format.to_vk() && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .unwrap_or_else(|| {
                    println!("Using default format for swapchain: {:?}", support.formats[0]);
                    support.formats[0]
                })
        };

        let extent = {
            if support.capabilities.current_extent.width != u32::MAX {
                support.capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: swapchain_description
                        .width
                        .clamp(support.capabilities.min_image_extent.width, support.capabilities.max_image_extent.width),
                    height: swapchain_description
                        .height
                        .clamp(support.capabilities.min_image_extent.height, support.capabilities.max_image_extent.height),
                }
            }
        };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle)
            .min_image_count(swapchain_description.frames_in_flight.clamp(support.capabilities.min_image_count, support.capabilities.max_image_count))
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(match old_swapchain {
                Some(s) => s.handle,
                None => vk::SwapchainKHR::null(),
            });

        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None).expect("Failed to create swapchain") };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain).expect("Failed to get swapchain images") };

        let inner_images: Vec<_> = images
            .iter()
            .map(|img| InnerImage {
                image: *img,
                mem_requirements: vk::MemoryRequirements::default(),
                format: swapchain_description.format.to_vk(),
                allocation: Allocation::default(),
            })
            .collect();

        let ids = inner_images
            .into_iter()
            .map(|img| {
                let inner_view = ctx.device.create_image_view(&img, &ImageViewDescription::default());
                let raw = inner_view.view;
                let view_id = ctx.image_views.write().unwrap().insert(inner_view);
                let image_id = ctx.images.write().unwrap().insert(img);

                Image {
                    default_view: ImageView {
                        raw: raw,
                        id: view_id,
                    },
                    id: image_id,
                }
            })
            .collect();

        return InnerSwapchain {
            handle: swapchain,
            swapchain_loader: swapchain_loader,
            images: ids,
            swapchain_description: *swapchain_description,
            acquire_semaphores: unsafe {
                (0..swapchain_description.frames_in_flight)
                    .map(|_| ctx.device.handle.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap())
                    .collect()
            },
            present_semaphores: unsafe {
                (0..swapchain_description.frames_in_flight)
                    .map(|_| ctx.device.handle.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap())
                    .collect()
            },
            fence: unsafe {
                (0..create_info.min_image_count)
                    .map(|_| ctx.device.handle.create_fence(&vk::FenceCreateInfo::default().flags(FenceCreateFlags::SIGNALED), None).unwrap())
                    .collect()
            },
            slot_used: vec![false; create_info.min_image_count as usize],
            frame_timeline: 0,
        };
    }

    pub(crate) fn acquire_image(&mut self) -> AcquiredImage {
        let slot = self.frame_timeline;

        let acquire_info = vk::AcquireNextImageInfoKHR::default()
            .swapchain(self.handle)
            .timeout(u64::MAX)
            .semaphore(self.acquire_semaphores[slot])
            .device_mask(1);

        let (index, _) = unsafe { self.swapchain_loader.acquire_next_image2(&acquire_info).expect("Failed to acquire image") };

        self.frame_timeline = (self.frame_timeline + 1) % self.swapchain_description.frames_in_flight as usize;

        return AcquiredImage {
            image: self.images[index as usize],
            image_index: index as usize,
            acquire_semaphore: self.acquire_semaphores[slot],
            present_semaphore: self.present_semaphores[slot],
        };
    }

    pub(crate) fn present(&mut self, accquired_image: &AcquiredImage, counter: Counter) {
        let ctx = crate::CONTEXT.get().unwrap();
        let (queue_id, value) = counter.decode();

        let queue = &ctx.queues[queue_id as usize];

        unsafe {
            ctx.device
                .handle
                .queue_submit2(
                    queue.queue,
                    &[
                        vk::SubmitInfo2::default()
                            .wait_semaphore_infos(&[
                                vk::SemaphoreSubmitInfo {
                                    semaphore: queue.semaphore,
                                    value,
                                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                                    device_index: 0,
                                    ..Default::default()
                                },
                            ])
                            .signal_semaphore_infos(&[
                                vk::SemaphoreSubmitInfo {
                                    semaphore: accquired_image.present_semaphore,
                                    value: 0,
                                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                                    device_index: 0,
                                    ..Default::default()
                                },
                            ]),
                    ],
                    vk::Fence::null(),
                )
                .unwrap();
        }

        unsafe {
            self.swapchain_loader
                .queue_present(
                    crate::CONTEXT.get().unwrap().queues[0].queue,
                    &vk::PresentInfoKHR::default()
                        .swapchains(&[self.handle])
                        .image_indices(&[accquired_image.image_index as u32])
                        .wait_semaphores(&[accquired_image.present_semaphore]),
                )
                .expect("Failed to preset image!!");
        }
    }

    pub(crate) fn cleanup(&self) {
        let ctx = crate::CONTEXT.get().unwrap();

        self.images.iter().for_each(|img| {
            ctx.images.write().unwrap().remove(img.id).unwrap();
            let view = ctx.image_views.write().unwrap().remove(img.default_view.id).unwrap();
            ctx.device.destroy_image_view(view);
        });

        unsafe {
            self.acquire_semaphores.iter().for_each(|s| ctx.device.handle.destroy_semaphore(*s, None));
            self.present_semaphores.iter().for_each(|s| ctx.device.handle.destroy_semaphore(*s, None));

            self.swapchain_loader.destroy_swapchain(self.handle, None);
        }
    }
} */

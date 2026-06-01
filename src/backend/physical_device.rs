use ash::vk;

use crate::backend::instance::Instance;

pub(crate) struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub transfer_family: Option<u32>,
    pub compute_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        return self.graphics_family.is_some() && self.compute_family.is_some() && self.transfer_family.is_some();
    }
}

pub(crate) struct PhysicalDevice {
    pub(crate) handle: vk::PhysicalDevice,
    pub(crate) queue_families: QueueFamilyIndices,
}

impl PhysicalDevice {
    pub(crate) fn select_physical_device(instance: &Instance, required_extensions: &[*const i8]) -> Option<PhysicalDevice> {
        let devices = unsafe { instance.handle.enumerate_physical_devices().expect("Failed to enumerate physical devices") };

        let mut best_device: Option<(i32, PhysicalDevice)> = None;

        for device in devices {
            let mut props: vk::PhysicalDeviceProperties2 = vk::PhysicalDeviceProperties2::default();
            unsafe {
                instance.handle.get_physical_device_properties2(device, &mut props);
            };

            if let Some(qf) = Self::get_queue_families(instance, device) {
                if !Self::check_device_extension_support(instance, device, required_extensions) {
                    continue;
                }

                // Score device: discrete = 1000, integrated = 100, others = 10
                let score = match props.properties.device_type {
                    ash::vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
                    ash::vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
                    _ => 10,
                };

                // Prefer larger max image dimension as tiebreaker
                let score = score + props.properties.limits.max_image_dimension2_d as i32;

                let candidate = PhysicalDevice {
                    handle: device,
                    queue_families: qf,
                };

                if let Some((best_score, _)) = &best_device {
                    if score > *best_score {
                        best_device = Some((score, candidate));
                    }
                } else {
                    best_device = Some((score, candidate));
                }
            }
        }

        return best_device.map(|(_, dev)| dev);
    }

    fn get_queue_families(instance: &Instance, physical_device: ash::vk::PhysicalDevice) -> Option<QueueFamilyIndices> {
        let queue_families = unsafe { instance.handle.get_physical_device_queue_family_properties(physical_device) };

        let mut indices = QueueFamilyIndices {
            graphics_family: None,
            transfer_family: None,
            compute_family: None,
        };

        for (i, family) in queue_families.iter().enumerate() {
            // Graphics
            if family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) && indices.graphics_family.is_none() {
                indices.graphics_family = Some(i as u32);
            }

            // Compute (dedicated if possible)
            if family.queue_flags.contains(ash::vk::QueueFlags::COMPUTE) && !family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) && indices.compute_family.is_none() {
                indices.compute_family = Some(i as u32);
            }

            // Transfer (dedicated if possible)
            if family.queue_flags.contains(ash::vk::QueueFlags::TRANSFER) && !family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) && !family.queue_flags.contains(ash::vk::QueueFlags::COMPUTE) && indices.transfer_family.is_none() {
                indices.transfer_family = Some(i as u32);
            }

            if indices.is_complete() {
                break;
            }
        }

        if indices.is_complete() {
            return Some(indices);
        } else {
            return None;
        }
    }

    fn check_device_extension_support(instance: &Instance, device: ash::vk::PhysicalDevice, required_extensions: &[*const i8]) -> bool {
        let available_extensions = unsafe { instance.handle.enumerate_device_extension_properties(device).expect("Failed to enumerate device extensions") };

        return required_extensions.iter().all(|&required_ptr| {
            let required_str = unsafe { std::ffi::CStr::from_ptr(required_ptr) };

            available_extensions.iter().any(|avail| {
                let avail_str = unsafe { std::ffi::CStr::from_ptr(avail.extension_name.as_ptr()) };

                avail_str == required_str
            })
        });
    }
}

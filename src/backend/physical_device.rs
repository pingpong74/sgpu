use ash::vk;
use smallvec::*;

use crate::backend::instance::Instance;

pub(crate) struct QueueFamilyIndices {
    pub(crate) graphics: Option<u32>,
    pub(crate) compute: Option<u32>,
    pub(crate) transfer: Option<u32>,
}

impl QueueFamilyIndices {
    pub(crate) fn get_familes(&self) -> SmallVec<[u32; 3]> {
        let mut families = smallvec![];

        self.graphics.map(|f| families.push(f));
        self.compute.map(|f| families.push(f));
        self.transfer.map(|f| families.push(f));

        return families;
    }

    fn is_complete(&self) -> bool { return self.graphics.is_some() && self.compute.is_some() && self.transfer.is_some(); }
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

                let score = {
                    let dev_score = match props.properties.device_type {
                        ash::vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
                        ash::vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
                        _ => 10,
                    };
                    let completeness_score = if qf.is_complete() { 100 } else { 10 };
                    completeness_score + dev_score
                };

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

        let mut graphics = None;
        let mut compute = None;
        let mut transfer = None;

        for (i, family) in queue_families.iter().enumerate() {
            let index = i as u32;

            // Graphics queue.
            if family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            // Prefer dedicated compute.
            if family.queue_flags.contains(ash::vk::QueueFlags::COMPUTE) && !family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) && compute.is_none() {
                compute = Some(index);
            }

            // Prefer dedicated transfer.
            if family.queue_flags.contains(ash::vk::QueueFlags::TRANSFER)
                && !family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS)
                && !family.queue_flags.contains(ash::vk::QueueFlags::COMPUTE)
                && transfer.is_none()
            {
                transfer = Some(index);
            }
        }

        return Some(QueueFamilyIndices {
            graphics,
            compute,
            transfer,
        });
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

use crate::api::SgpuInititizationInfo;

use ash::vk;
use raw_window_handle;

pub(crate) struct Instance {
    pub(crate) handle: ash::Instance,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    debug_loader: Option<ash::ext::debug_utils::Instance>,
}

impl Instance {
    pub(crate) fn new(sgpu_init_info: &SgpuInititizationInfo) -> Instance {
        let entry = ash::Entry::linked();

        let mut required_extensions = vec![ash::khr::surface::NAME.as_ptr()];
        let supported_exts = unsafe { entry.enumerate_instance_extension_properties(None).unwrap() };
        let supported_names: Vec<&std::ffi::CStr> = supported_exts.iter().map(|e| unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }).collect();

        let mut push_if_supported = |ext_name: &std::ffi::CStr| {
            if supported_names.contains(&ext_name) {
                required_extensions.push(ext_name.as_ptr());
                return true;
            }
            false
        };

        if let Some(raw_window_handle) = sgpu_init_info.window_handle {
            match raw_window_handle {
                raw_window_handle::RawWindowHandle::Win32(_) => {
                    if !push_if_supported(ash::khr::win32_surface::NAME) {
                        println!("Warning: Win32 surface extension not supported by Vulkan driver/layer");
                    }
                }
                raw_window_handle::RawWindowHandle::Wayland(_) => {
                    if !push_if_supported(ash::khr::wayland_surface::NAME) {
                        println!("Warning: Wayland surface extension not supported by Vulkan driver/layer");
                    }
                }
                raw_window_handle::RawWindowHandle::Xcb(_) => {
                    if !push_if_supported(ash::khr::xcb_surface::NAME) {
                        println!("Warning: Xcb surface extension not supported by Vulkan driver/layer");
                    }
                }
                raw_window_handle::RawWindowHandle::Xlib(_) => {
                    if !push_if_supported(ash::khr::xlib_surface::NAME) {
                        println!("Warning: Xlib surface extension not supported by Vulkan driver/layer");
                    }
                }
                raw_window_handle::RawWindowHandle::AppKit(_) => {
                    if !push_if_supported(ash::ext::metal_surface::NAME) {
                        println!("Warning: Metal surface extension not supported by Vulkan driver/layer");
                    }
                }
                _ => {}
            }
        }

        if sgpu_init_info.enable_validation_layers {
            required_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let app_info = vk::ApplicationInfo {
            api_version: vk::API_VERSION_1_3,
            ..Default::default()
        };

        let mut create_info = vk::InstanceCreateInfo::default().application_info(&app_info).enabled_extension_names(&required_extensions);

        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
            .pfn_user_callback(Some(Instance::vulkan_debug_callback));

        if sgpu_init_info.enable_validation_layers {
            create_info = create_info.push_next(&mut debug_create_info);
        }

        let instance = unsafe { entry.create_instance(&create_info, None).expect("Failed to create instance") };

        let mut debug_messenger: Option<vk::DebugUtilsMessengerEXT> = None;
        let mut debug_loader: Option<ash::ext::debug_utils::Instance> = None;

        if sgpu_init_info.enable_validation_layers {
            let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);

            debug_messenger = Some(unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_create_info, None) }.expect("Debug Utils Messenger creation failed"));

            debug_loader = Some(debug_utils_loader);
        }

        return Instance {
            handle: instance,
            debug_messenger: debug_messenger,
            debug_loader: debug_loader,
        };
    }

    pub(crate) fn cleanup(&mut self) {
        unsafe {
            if !self.debug_messenger.is_none() {
                if self.debug_loader.is_none() {
                    panic!("Created debug utils but not debug loader")
                }

                self.debug_loader.as_mut().unwrap().destroy_debug_utils_messenger(self.debug_messenger.unwrap(), None);
            }

            self.handle.destroy_instance(None);
        };
    }

    #[allow(unused)]
    unsafe extern "system" fn vulkan_debug_callback(
        severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
        types: ash::vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
        _user: *mut std::ffi::c_void,
    ) -> ash::vk::Bool32 {
        let message = unsafe { std::ffi::CStr::from_ptr((*data).p_message).to_string_lossy().into_owned() };
        println!("[VULKAN, {:?} {:?}]: {}", severity, types, message);

        ash::vk::FALSE
    }
}

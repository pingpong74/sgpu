#![no_std]
#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]

use core::marker::PhantomData;

use spirv_std::arch;
use spirv_std::byte_addressable_buffer::ByteAddressableBuffer;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::{RuntimeArray, TypedBuffer};

pub use spirv_std::image::{Cubemap, Image1d, Image1dI, Image1dU, Image2d, Image2dI, Image2dU, Image3d, Image3dI, Image3dU, StorageImage1d, StorageImage1dI, StorageImage1dU, StorageImage2d, StorageImage2dI, StorageImage2dU, StorageImage3d, StorageImage3dI, StorageImage3dU};
pub use spirv_std::{ByteAddressableBuffer as ByteAddressBuffer, glam, spirv};

pub const DESCRIPTOR_SET: u32 = 0;
pub const SAMPLED_IMAGE_BINDING: u32 = 0;
pub const STORAGE_IMAGE_BINDING: u32 = 1;
pub const BINDLESS_BUFFER_BINDING: u32 = 2;
pub const MAX_PUSH_CONSTANT_SIZE: usize = 256;

pub type BindlessBuffers = RuntimeArray<TypedBuffer<[u32]>>;

const ATOMIC_SCOPE: u32 = Scope::Device as u32;
const ATOMIC_SEMANTICS: u32 = Semantics::NONE.bits();

pub struct ReadOnlyBuffer<'a, T> {
    data: &'a [u32],
    _marker: PhantomData<T>,
}

impl<T> ReadOnlyBuffer<'_, T> {
    #[inline]
    pub fn get(handle: u32) -> Self {
        let buffer = unsafe { bindless_buffers().index(handle as usize) };
        ReadOnlyBuffer {
            data: buffer,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn load(&self, byte_offset: u32) -> T { unsafe { ByteAddressableBuffer::from_slice(self.data).load(byte_offset) } }

    #[inline]
    pub fn load_indexed(&self, index: u32) -> T { self.load(index * core::mem::size_of::<T>() as u32) }
}

pub struct ReadWriteBuffer<'a, T> {
    data: &'a mut [u32],
    _marker: PhantomData<T>,
}

impl<T> ReadWriteBuffer<'_, T> {
    #[inline]
    pub fn get(handle: u32) -> Self {
        let buffer = unsafe { bindless_buffers_mut().index_mut(handle as usize) };
        ReadWriteBuffer {
            data: buffer,
            _marker: PhantomData,
        }
    }
    #[inline]
    pub fn load(&self, byte_offset: u32) -> T { unsafe { ByteAddressableBuffer::from_slice(self.data).load(byte_offset) } }

    #[inline]
    pub fn load_indexed(&self, index: u32) -> T { self.load(index * core::mem::size_of::<T>() as u32) }

    #[inline]
    pub fn store(&mut self, value: T, byte_offset: u32) {
        let mut buffer = ByteAddressableBuffer::from_mut_slice(&mut *self.data);
        unsafe { buffer.store(byte_offset, value) }
    }

    #[inline]
    pub fn store_indexed(&mut self, value: T, index: u32) { self.store(value, index * core::mem::size_of::<T>() as u32) }
}

impl ReadWriteBuffer<'_, u32> {
    #[inline]
    pub fn interlocked_add(&mut self, byte_offset: u32, value: u32) -> u32 { unsafe { arch::atomic_i_add::<u32, ATOMIC_SCOPE, ATOMIC_SEMANTICS>(self.word(byte_offset), value) } }

    #[inline]
    pub fn interlocked_min(&mut self, byte_offset: u32, value: u32) -> u32 { unsafe { arch::atomic_u_min::<u32, ATOMIC_SCOPE, ATOMIC_SEMANTICS>(self.word(byte_offset), value) } }

    #[inline]
    pub fn interlocked_max(&mut self, byte_offset: u32, value: u32) -> u32 { unsafe { arch::atomic_u_max::<u32, ATOMIC_SCOPE, ATOMIC_SEMANTICS>(self.word(byte_offset), value) } }

    #[inline]
    pub fn interlocked_compare_exchange(&mut self, byte_offset: u32, compare: u32, value: u32) -> u32 { unsafe { arch::atomic_compare_exchange::<u32, ATOMIC_SCOPE, ATOMIC_SEMANTICS, ATOMIC_SEMANTICS>(self.word(byte_offset), value, compare) } }

    fn word(&mut self, byte_offset: u32) -> &mut u32 {
        assert!(byte_offset.is_multiple_of(4), "atomic byte offset must be 4-byte aligned");
        &mut self.data[(byte_offset / 4) as usize]
    }
}

impl ReadWriteBuffer<'_, i32> {
    #[inline]
    pub fn interlocked_add(&mut self, byte_offset: u32, value: i32) -> i32 { unsafe { arch::atomic_i_add::<i32, ATOMIC_SCOPE, ATOMIC_SEMANTICS>(self.word(byte_offset), value) } }

    #[inline]
    pub fn interlocked_min(&mut self, byte_offset: u32, value: i32) -> i32 { unsafe { arch::atomic_s_min::<i32, ATOMIC_SCOPE, ATOMIC_SEMANTICS>(self.word(byte_offset), value) } }

    #[inline]
    pub fn interlocked_max(&mut self, byte_offset: u32, value: i32) -> i32 { unsafe { arch::atomic_s_max::<i32, ATOMIC_SCOPE, ATOMIC_SEMANTICS>(self.word(byte_offset), value) } }

    #[inline]
    pub fn interlocked_compare_exchange(&mut self, byte_offset: u32, compare: i32, value: i32) -> i32 { unsafe { arch::atomic_compare_exchange::<i32, ATOMIC_SCOPE, ATOMIC_SEMANTICS, ATOMIC_SEMANTICS>(self.word(byte_offset), value, compare) } }

    fn word(&mut self, byte_offset: u32) -> &mut i32 {
        assert!(byte_offset.is_multiple_of(4), "atomic byte offset must be 4-byte aligned");
        unsafe { &mut *(&mut self.data[(byte_offset / 4) as usize] as *mut u32).cast() }
    }
}

// # Safety:
// Usage of raw spriv
pub unsafe fn bindless_buffers() -> &'static BindlessBuffers {
    #[cfg(target_arch = "spirv")]
    {
        let mut slot = core::mem::MaybeUninit::<&BindlessBuffers>::uninit();
        unsafe {
            core::arch::asm!(
                "%var = OpVariable typeof*{slot} StorageBuffer",
                "OpDecorate %var DescriptorSet {descriptor_set}",
                "OpDecorate %var Binding {binding}",
                "OpDecorate %var NonWritable",
                "OpStore {slot} %var",
                descriptor_set = const DESCRIPTOR_SET,
                binding = const BINDLESS_BUFFER_BINDING,
                slot = in(reg) slot.as_mut_ptr(),
            );
        }
        unsafe { slot.assume_init() }
    }
    #[cfg(not(target_arch = "spirv"))]
    unimplemented!()
}

// # Safety:
// Usage of raw spriv
pub unsafe fn bindless_buffers_mut() -> &'static mut BindlessBuffers {
    #[cfg(target_arch = "spirv")]
    {
        let mut slot = core::mem::MaybeUninit::<&mut BindlessBuffers>::uninit();
        unsafe {
            core::arch::asm!(
                "%var = OpVariable typeof*{slot} StorageBuffer",
                "OpDecorate %var DescriptorSet {descriptor_set}",
                "OpDecorate %var Binding {binding}",
                "OpStore {slot} %var",
                descriptor_set = const DESCRIPTOR_SET,
                binding = const BINDLESS_BUFFER_BINDING,
                slot = in(reg) slot.as_mut_ptr(),
            );
        }
        unsafe { slot.assume_init() }
    }
    #[cfg(not(target_arch = "spirv"))]
    unimplemented!()
}

macro_rules! image_getters {
    ($binding:expr; $(fn $name:ident($ty:ty)),+ $(,)?) => {
        $(
            #[inline(never)]
            pub fn $name(handle: u32) -> &'static $ty {
                #[cfg(target_arch = "spirv")]
                {
                    let mut slot = core::mem::MaybeUninit::<&RuntimeArray<$ty>>::uninit();
                    unsafe {
                        core::arch::asm!(
                            "%var = OpVariable typeof*{slot} UniformConstant",
                            "OpDecorate %var DescriptorSet {descriptor_set}",
                            "OpDecorate %var Binding {binding}",
                            "OpStore {slot} %var",
                            descriptor_set = const DESCRIPTOR_SET,
                            binding = const $binding,
                            slot = in(reg) slot.as_mut_ptr(),
                        );
                    }
                    let images = unsafe { slot.assume_init() };
                    unsafe { images.index(handle as usize) }
                }
                #[cfg(not(target_arch = "spirv"))]
                {
                    let _ = handle;
                    unimplemented!()
                }
            }
        )+
    };
}

image_getters! {
    SAMPLED_IMAGE_BINDING;
    fn get_sampled_1d_f(Image1d),
    fn get_sampled_2d_f(Image2d),
    fn get_sampled_3d_f(Image3d),
    fn get_sampled_cube_f(Cubemap),
    fn get_sampled_1d_u(Image1dU),
    fn get_sampled_2d_u(Image2dU),
    fn get_sampled_3d_u(Image3dU),
    fn get_sampled_1d_i(Image1dI),
    fn get_sampled_2d_i(Image2dI),
    fn get_sampled_3d_i(Image3dI),
}

image_getters! {
    STORAGE_IMAGE_BINDING;
    fn get_rw_1d_f(StorageImage1d),
    fn get_rw_2d_f(StorageImage2d),
    fn get_rw_3d_f(StorageImage3d),
    fn get_rw_1d_u(StorageImage1dU),
    fn get_rw_2d_u(StorageImage2dU),
    fn get_rw_3d_u(StorageImage3dU),
    fn get_rw_1d_i(StorageImage1dI),
    fn get_rw_2d_i(StorageImage2dI),
    fn get_rw_3d_i(StorageImage3dI),
}

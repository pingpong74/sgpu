#![no_std]
#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]

use core::marker::PhantomData;

use spirv_std::arch;
use spirv_std::byte_addressable_buffer::ByteAddressableBuffer;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::{RuntimeArray, Sampler, TypedBuffer};

use spirv_std::image::{Cubemap, Image1d, Image1dI, Image1dU, Image2d, Image2dI, Image2dU, Image3d, Image3dI, Image3dU, StorageImage1d, StorageImage1dI, StorageImage1dU, StorageImage2d, StorageImage2dI, StorageImage2dU, StorageImage3d, StorageImage3dI, StorageImage3dU};

pub const DESCRIPTOR_SET: u32 = 0;
pub const SAMPLED_IMAGE_BINDING: u32 = 0;
pub const STORAGE_IMAGE_BINDING: u32 = 1;
pub const BINDLESS_BUFFER_BINDING: u32 = 2;
pub const SAMPLER_BINDING: u32 = 3;
pub const MAX_PUSH_CONSTANT_SIZE: usize = 256;

pub type BindlessBuffers = RuntimeArray<TypedBuffer<[u32]>>;
pub type BindlessSamplers = RuntimeArray<Sampler>;

const ATOMIC_SCOPE: u32 = Scope::Device as u32;
const ATOMIC_SEMANTICS: u32 = Semantics::NONE.bits();

pub trait Usage {
    const WRITEABLE: bool;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Read;

impl Usage for Read {
    const WRITEABLE: bool = false;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Write;

impl Usage for Write {
    const WRITEABLE: bool = true;
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BufferId<U: Usage> {
    id: u32,
    _u: PhantomData<U>,
}

impl<U: Usage> BufferId<U> {
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self {
            id,
            _u: PhantomData,
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StorageImageId<U: Usage> {
    id: u32,
    _u: PhantomData<U>,
}

impl<U: Usage> StorageImageId<U> {
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self {
            id,
            _u: PhantomData,
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SampledImageId {
    id: u32,
}

impl SampledImageId {
    #[inline]
    pub const fn new(id: u32) -> Self { Self { id } }
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SamplerId {
    id: u32,
}

impl SamplerId {
    #[inline]
    pub const fn new(id: u32) -> Self { Self { id } }

    pub fn get_sampler(id: SamplerId) -> &'static Sampler {
        #[cfg(target_arch = "spirv")]
        {
            let mut slot = core::mem::MaybeUninit::<&RuntimeArray<Sampler>>::uninit();
            unsafe {
                core::arch::asm!(
                    "%var = OpVariable typeof*{slot} UniformConstant",
                    "OpDecorate %var DescriptorSet {descriptor_set}",
                    "OpDecorate %var Binding {binding}",
                    "OpStore {slot} %var",
                    descriptor_set = const DESCRIPTOR_SET,
                    binding = const SAMPLER_BINDING,
                    slot = in(reg) slot.as_mut_ptr(),
                );
            }
            let samplers = unsafe { slot.assume_init() };
            unsafe { samplers.index(id.id as usize) }
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            let _ = id;
            unimplemented!()
        }
    }
}

pub struct ReadOnlyBuffer<'a, T> {
    data: &'a [u32],
    _marker: PhantomData<T>,
}

impl<T> ReadOnlyBuffer<'_, T> {
    #[inline]
    pub fn get(id: BufferId<Read>) -> Self {
        let buffer = unsafe { bindless_buffers().index(id.id as usize) };
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
    pub fn get(id: BufferId<Write>) -> Self {
        let buffer = unsafe { bindless_buffers_mut().index_mut(id.id as usize) };
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
    ($binding:expr, $id:ty, non_writable; $(fn $name:ident($ty:ty)),+ $(,)?) => {
        $(
            pub fn $name(id: $id) -> &'static $ty {
                image_getters!(@body id, $id, $ty, $binding, "OpDecorate %var NonWritable")
            }
        )+
    };
    ($binding:expr, $id:ty; $(fn $name:ident($ty:ty)),+ $(,)?) => {
        $(
            pub fn $name(id: $id) -> &'static $ty {
                image_getters!(@body id, $id, $ty, $binding,)
            }
        )+
    };
    (@body $pid:ident, $id:ty, $ty:ty, $binding:expr, $($decor:literal)?) => {
        {
            #[cfg(target_arch = "spirv")]
            {
                let mut slot = core::mem::MaybeUninit::<&RuntimeArray<$ty>>::uninit();
                unsafe {
                    core::arch::asm!(
                        "%var = OpVariable typeof*{slot} UniformConstant",
                        "OpDecorate %var DescriptorSet {descriptor_set}",
                        "OpDecorate %var Binding {binding}",
                        $($decor,)?
                        "OpStore {slot} %var",
                        descriptor_set = const DESCRIPTOR_SET,
                        binding = const $binding,
                        slot = in(reg) slot.as_mut_ptr(),
                    );
                }
                let images = unsafe { slot.assume_init() };
                unsafe { images.index($pid.id as usize) }
            }
            #[cfg(not(target_arch = "spirv"))]
            {
                let _ = $pid;
                unimplemented!()
            }
        }
    };
}

// Sampled 1D array
pub type Image1dArray = spirv_std::Image!(1D, type = f32, sampled, arrayed);
pub type Image1dArrayU = spirv_std::Image!(1D, type = u32, sampled, arrayed);
pub type Image1dArrayI = spirv_std::Image!(1D, type = i32, sampled, arrayed);

// Sampled 2D array
pub type Image2dArray = spirv_std::Image!(2D, type = f32, sampled, arrayed);
pub type Image2dArrayU = spirv_std::Image!(2D, type = u32, sampled, arrayed);
pub type Image2dArrayI = spirv_std::Image!(2D, type = i32, sampled, arrayed);

// Sampled Cube arrayed
pub type CubemapArray = spirv_std::Image!(cube, type = f32, sampled, arrayed);
pub type CubemapArrayU = spirv_std::Image!(cube, type = u32, sampled, arrayed);
pub type CubemapArrayI = spirv_std::Image!(cube, type = i32, sampled, arrayed);

// Storage 1D array
pub type StorageImage1dArray = spirv_std::image::Image!(1D, type = f32, storage, arrayed);
pub type StorageImage1dArrayU = spirv_std::image::Image!(1D, type = u32, storage, arrayed);
pub type StorageImage1dArrayI = spirv_std::image::Image!(1D, type = i32, storage, arrayed);

// Storage 2D array
pub type StorageImage2dArray = spirv_std::image::Image!(2D, type = f32, storage, arrayed);
pub type StorageImage2dArrayU = spirv_std::image::Image!(2D, type = u32, storage, arrayed);
pub type StorageImage2dArrayI = spirv_std::image::Image!(2D, type = i32, storage, arrayed);

image_getters! {
    SAMPLED_IMAGE_BINDING, SampledImageId;
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
    // arrays
    fn get_sampled_1d_array_f(Image1dArray),
    fn get_sampled_1d_array_u(Image1dArrayU),
    fn get_sampled_1d_array_i(Image1dArrayI),
    fn get_sampled_2d_array_f(Image2dArray),
    fn get_sampled_2d_array_u(Image2dArrayU),
    fn get_sampled_2d_array_i(Image2dArrayI),
    fn get_sampled_cube_array_f(CubemapArray),
    fn get_sampled_cube_array_u(CubemapArrayU),
    fn get_sampled_cube_array_i(CubemapArrayI),
}

image_getters! {
    STORAGE_IMAGE_BINDING, StorageImageId<Read>, non_writable;
    fn get_ro_1d_f(StorageImage1d),
    fn get_ro_2d_f(StorageImage2d),
    fn get_ro_3d_f(StorageImage3d),
    fn get_ro_1d_u(StorageImage1dU),
    fn get_ro_2d_u(StorageImage2dU),
    fn get_ro_3d_u(StorageImage3dU),
    fn get_ro_1d_i(StorageImage1dI),
    fn get_ro_2d_i(StorageImage2dI),
    fn get_ro_3d_i(StorageImage3dI),

    // array
    fn get_ro_1d_array_f(StorageImage1dArray),
    fn get_ro_1d_array_u(StorageImage1dArrayU),
    fn get_ro_1d_array_i(StorageImage1dArrayI),
    fn get_ro_2d_array_f(StorageImage2dArray),
    fn get_ro_2d_array_u(StorageImage2dArrayU),
    fn get_ro_2d_array_i(StorageImage2dArrayI),
}

image_getters! {
    STORAGE_IMAGE_BINDING, StorageImageId<Write>;
    fn get_rw_1d_f(StorageImage1d),
    fn get_rw_2d_f(StorageImage2d),
    fn get_rw_3d_f(StorageImage3d),
    fn get_rw_1d_u(StorageImage1dU),
    fn get_rw_2d_u(StorageImage2dU),
    fn get_rw_3d_u(StorageImage3dU),
    fn get_rw_1d_i(StorageImage1dI),
    fn get_rw_2d_i(StorageImage2dI),
    fn get_rw_3d_i(StorageImage3dI),
    // arrayed
    fn get_rw_1d_array_f(StorageImage1dArray),
    fn get_rw_1d_array_u(StorageImage1dArrayU),
    fn get_rw_1d_array_i(StorageImage1dArrayI),
    fn get_rw_2d_array_f(StorageImage2dArray),
    fn get_rw_2d_array_u(StorageImage2dArrayU),
    fn get_rw_2d_array_i(StorageImage2dArrayI),
}

// Draw indirect command
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

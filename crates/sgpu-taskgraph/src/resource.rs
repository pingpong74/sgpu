use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::access::{BufferAccess, ImageAccess};

pub struct Id<T> {
    pub(crate) index: u32,
    _m: PhantomData<fn() -> T>,
}

impl<T> Copy for Id<T> {}
impl<T> Clone for Id<T> {
    fn clone(&self) -> Self { *self }
}
impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool { self.index == other.index }
}
impl<T> Eq for Id<T> {}
impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> Ordering { self.index.cmp(&other.index) }
}
impl<T> Hash for Id<T> {
    fn hash<H: Hasher>(&self, state: &mut H) { self.index.hash(state); }
}
impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("Id").field("index", &self.index).finish() }
}

impl<T> Id<T> {
    pub(crate) fn new(index: u32) -> Self {
        Id {
            index,
            _m: PhantomData,
        }
    }

    pub fn index(self) -> u32 { self.index }
}

pub struct PassImage<A: ImageAccess> {
    pub(crate) index: u32,
    _m: PhantomData<fn() -> A>,
}

impl<A: ImageAccess> Copy for PassImage<A> {}
impl<A: ImageAccess> Clone for PassImage<A> {
    fn clone(&self) -> Self { *self }
}
impl<A: ImageAccess> PartialEq for PassImage<A> {
    fn eq(&self, other: &Self) -> bool { self.index == other.index }
}
impl<A: ImageAccess> Eq for PassImage<A> {}
impl<A: ImageAccess> fmt::Debug for PassImage<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("PassImage").field("index", &self.index).finish() }
}

impl<A: ImageAccess> PassImage<A> {
    pub(crate) fn new(index: u32) -> Self {
        PassImage {
            index,
            _m: PhantomData,
        }
    }
}

pub struct PassBuffer<A: BufferAccess> {
    pub(crate) index: u32,
    _m: PhantomData<fn() -> A>,
}

impl<A: BufferAccess> Copy for PassBuffer<A> {}
impl<A: BufferAccess> Clone for PassBuffer<A> {
    fn clone(&self) -> Self { *self }
}
impl<A: BufferAccess> PartialEq for PassBuffer<A> {
    fn eq(&self, other: &Self) -> bool { self.index == other.index }
}
impl<A: BufferAccess> Eq for PassBuffer<A> {}
impl<A: BufferAccess> fmt::Debug for PassBuffer<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("PassBuffer").field("index", &self.index).finish() }
}

impl<A: BufferAccess> PassBuffer<A> {
    pub(crate) fn new(index: u32) -> Self {
        PassBuffer {
            index,
            _m: PhantomData,
        }
    }
}

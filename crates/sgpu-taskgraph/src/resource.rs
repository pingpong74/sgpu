use std::fmt;
use std::marker::PhantomData;

use crate::access::{BufferAccess, ImageAccess};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PassImageId(pub(crate) u32, u32);

impl PassImageId {
    pub(crate) fn new(index: u32, graph: u32) -> Self { Self(index, graph) }

    pub(crate) fn check_graph(&self, graph_id: u32) -> u32 {
        assert!(self.1 == graph_id, "id {self:?} does not belong to this graph");
        self.0
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PassBufferId(pub(crate) u32, u32);

impl PassBufferId {
    pub(crate) fn new(index: u32, graph: u32) -> Self { Self(index, graph) }

    pub(crate) fn check_graph(&self, graph_id: u32) -> u32 {
        assert!(self.1 == graph_id, "id {self:?} does not belong to this graph");
        self.0
    }
}

/// Graph Image with its Access Type
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PassImage<A: ImageAccess> {
    pub(crate) index: u32,
    _m: PhantomData<fn() -> A>,
}

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

/// Graph Buffer with its Access Type
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PassBuffer<A: BufferAccess> {
    pub(crate) index: u32,
    _m: PhantomData<fn() -> A>,
}

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

/// Graph image view with its Access Type
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PassView<A: ImageAccess> {
    pub(crate) index: u32,
    _m: PhantomData<fn() -> A>,
}

impl<A: ImageAccess> fmt::Debug for PassView<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("PassView").field("index", &self.index).finish() }
}

impl<A: ImageAccess> PassView<A> {
    pub(crate) fn new(index: u32) -> Self {
        PassView {
            index,
            _m: PhantomData,
        }
    }
}

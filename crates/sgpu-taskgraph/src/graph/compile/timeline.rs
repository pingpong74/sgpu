use std::collections::HashMap;

use sgpu::{AccessType, PipelineStage};

use crate::TaskGraph;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Resource {
    Image(u32),
    Buffer(u32),
}

#[derive(Debug)]
pub(crate) struct AccessGroup {
    pub(crate) accesses: Vec<AccessType>,
    pub(crate) members: Vec<usize>,
}

impl AccessGroup {
    pub(crate) fn is_present(&self) -> bool { self.accesses.len() == 1 && self.accesses[0] == AccessType::Present && self.members.len() == 1 }
}

#[derive(Debug)]
pub(crate) struct Timeline {
    pub(crate) resource: Resource,
    pub(crate) transient: bool,
    pub(crate) groups: Vec<AccessGroup>,
}

impl Timeline {
    fn push_group(&mut self, access: AccessType, node: usize) {
        let mergeable = self.groups.last().is_some_and(|last| !conflicts(&last.accesses, &[access]));

        if mergeable {
            let last = self.groups.last_mut().unwrap();
            last.accesses.push(access);
            last.members.push(node);
        } else {
            self.groups.push(AccessGroup {
                accesses: vec![access],
                members: vec![node],
            });
        }
    }

    pub(crate) fn validate_present(&self) {
        for (gi, group) in self.groups.iter().enumerate() {
            if !is_present(&group.accesses) {
                continue;
            }
            assert!(gi == self.groups.len() - 1, "{:?} is accessed after it was left in the Present state", self.resource);
            assert!(gi > 0, "{:?} is presented without ever being written", self.resource);
        }
    }
}

impl<'g, R, D> TaskGraph<'g, R, D> {
    pub(crate) fn build_timelines(&self, declared: &[super::NodeDeclaration<'g, R, D>]) -> Vec<Timeline> {
        let mut timelines: Vec<Timeline> = Vec::new();
        let mut res_to_timeline = HashMap::new();

        for (node, node_declaration) in declared.iter().enumerate() {
            for &(img, access) in &node_declaration.images {
                let ti = res_to_timeline.entry(Resource::Image(img)).or_insert_with(|| {
                    timelines.push(Timeline {
                        resource: Resource::Image(img),
                        transient: self.images[img as usize].is_transient(),
                        groups: Vec::new(),
                    });
                    timelines.len() - 1
                });
                timelines[*ti].push_group(access, node);
            }

            for &(buf, access) in &node_declaration.buffers {
                let ti = res_to_timeline.entry(Resource::Buffer(buf)).or_insert_with(|| {
                    timelines.push(Timeline {
                        resource: Resource::Buffer(buf),
                        transient: self.buffers[buf as usize].is_transient(),
                        groups: Vec::new(),
                    });
                    timelines.len() - 1
                });
                timelines[*ti].push_group(access, node);
            }
        }

        timelines
    }
}

pub(super) fn is_present(accesses: &[AccessType]) -> bool { accesses.iter().any(|a| *a == AccessType::Present) }

pub(super) fn is_write(accesses: &[AccessType]) -> bool { accesses.iter().any(|a| a.is_write()) }

pub(super) fn stage_of(accesses: &[AccessType]) -> PipelineStage { accesses.iter().fold(PipelineStage::NONE, |acc, &a| acc | a.stage()) }

pub(super) fn conflicts(a: &[AccessType], b: &[AccessType]) -> bool { is_write(a) || is_write(b) || a[0].layout_differs(b[0]) || is_present(a) || is_present(b) }

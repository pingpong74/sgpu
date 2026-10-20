use sgpu::{AccessType, PipelineStage, QueueType};

use crate::TaskGraph;

pub(super) enum Resource {
    Image(u32),
    Buffer(u32),
}

pub(super) struct AccessGroup {
    pub(super) accesses: Vec<AccessType>,
    pub(super) members: Vec<usize>,
    // Only used by the member present access
    pub(super) location: Option<(usize, QueueType)>,
}

impl AccessGroup {
    pub(super) fn is_present(&self) -> bool { self.members.is_empty() }
}

pub(super) struct Timeline {
    pub(super) resource: Resource,
    pub(super) groups: Vec<AccessGroup>,
}

impl<'g, R, D> TaskGraph<'g, R, D> {
    pub(super) fn build_timelines(&self, declared: &[super::compile::NodeDeclaration<'g, R, D>]) -> Vec<Timeline> {
        let mut timelines: Vec<Timeline> = Vec::new();

        for (node, node_declaration) in declared.iter().enumerate() {
            for &(img, access) in &node_declaration.images {
                let ti = match timelines.iter().position(|tl| matches!(tl.resource, Resource::Image(i) if i == img)) {
                    Some(i) => i,
                    None => {
                        timelines.push(Timeline {
                            resource: Resource::Image(img),
                            groups: Vec::new(),
                        });
                        timelines.len() - 1
                    }
                };
                push_group(&mut timelines[ti], access, node);
            }

            for &(buf, access) in &node_declaration.buffers {
                let ti = match timelines.iter().position(|tl| matches!(tl.resource, Resource::Buffer(b) if b == buf)) {
                    Some(i) => i,
                    None => {
                        timelines.push(Timeline {
                            resource: Resource::Buffer(buf),
                            groups: Vec::new(),
                        });
                        timelines.len() - 1
                    }
                };
                push_group(&mut timelines[ti], access, node);
            }
        }

        timelines
    }

    pub(super) fn append_present(&self, timelines: &mut [Timeline]) -> Option<usize> {
        let id = self.present_id?;
        let ti = timelines.iter().position(|tl| matches!(tl.resource, Resource::Image(i) if i == id.index)).expect("graph.present_transition: image is never accessed by any node");

        let tl = &mut timelines[ti];
        tl.groups.push(AccessGroup {
            accesses: vec![AccessType::Present],
            members: Vec::new(),
            location: None,
        });
        Some(ti)
    }
}

pub(super) fn is_write(accesses: &[AccessType]) -> bool { accesses.iter().any(|a| a.is_write()) }

pub(super) fn stage_of(accesses: &[AccessType]) -> PipelineStage { accesses.iter().fold(PipelineStage::NONE, |acc, &a| acc | a.stage()) }

pub(super) fn conflicts(a: &[AccessType], b: &[AccessType]) -> bool { is_write(a) || is_write(b) || a[0].layout_differs(b[0]) }

pub(super) fn push_group(timeline: &mut Timeline, access: AccessType, node: usize) {
    let mergeable = timeline.groups.last().is_some_and(|last| !is_write(&last.accesses) && !access.is_write() && !access.layout_differs(last.accesses[0]));

    if mergeable {
        let last = timeline.groups.last_mut().unwrap();
        last.accesses.push(access);
        last.members.push(node);
    } else {
        timeline.groups.push(AccessGroup {
            accesses: vec![access],
            members: vec![node],
            location: None,
        });
    }
}

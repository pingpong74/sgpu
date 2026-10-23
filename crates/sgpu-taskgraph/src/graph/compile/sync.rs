//! Builds the per-queue submission plan: waits, barriers, and record order.

use std::collections::HashSet;

use super::{
    QUEUES,
    alias::{AliasSchedule, Cycle},
    schedule::Schedule,
    timeline::{Resource, Timeline, conflicts, is_write, stage_of},
};
use crate::TaskGraph;

use sgpu::{AccessType, PipelineStage, QueueType};

#[derive(Clone, Debug)]
pub(crate) struct CompiledImageBarrier {
    pub(crate) previous: Vec<AccessType>,
    pub(crate) next: Vec<AccessType>,
    pub(crate) discard: bool,
    pub(crate) image: u32,
    pub(crate) is_graph_owned: bool,
}

#[derive(Default, Debug)]
pub(crate) struct CompiledGlobalBarrier {
    pub(crate) previous: Vec<AccessType>,
    pub(crate) next: Vec<AccessType>,
}

#[derive(Default, Debug)]
pub(crate) struct CompiledBarriers {
    pub(crate) images: Vec<CompiledImageBarrier>,
    pub(crate) global: CompiledGlobalBarrier,
}

impl CompiledBarriers {
    fn is_empty(&self) -> bool { self.images.is_empty() && self.global.previous.is_empty() && self.global.next.is_empty() }

    fn add_global(&mut self, previous: &[AccessType], next: &[AccessType]) {
        extend_unique(&mut self.global.previous, previous);
        extend_unique(&mut self.global.next, next);
    }

    fn absorb(&mut self, other: &CompiledBarriers) {
        self.images.extend_from_slice(&other.images);
        self.add_global(&other.global.previous, &other.global.next);
    }
}

fn extend_unique(into: &mut Vec<AccessType>, from: &[AccessType]) {
    for &access in from {
        if !into.contains(&access) {
            into.push(access);
        }
    }
}

/// One submission.
#[derive(Debug)]
pub(crate) struct Cell {
    pub(crate) queue: QueueType,
    pub(crate) wave: usize,
    pub(crate) nodes: Vec<usize>,
    /// (source queue, source cell, stage)
    pub(crate) waits: Vec<(QueueType, usize, PipelineStage)>,
    /// Previous-cycle waits on the source queue's last counter.
    pub(crate) slot_waits: Vec<(QueueType, PipelineStage)>,
    pub(crate) acquire: bool,
    pub(crate) terminal: bool,
    pub(crate) undefined_barriers: CompiledBarriers,
    pub(crate) barriers: CompiledBarriers,
}

/// Per-queue cells and the order in which the executor records them.
#[derive(Debug)]
pub(crate) struct CompiledSync {
    pub(crate) cells: [Vec<Cell>; 3],
    pub(crate) order: Vec<(QueueType, usize)>,
}

fn qi(queue: QueueType) -> usize { QUEUES.iter().position(|&q| q == queue).expect("node queue is a graph queue") }

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SliceId {
    batch: usize,
    queue: usize,
}

/// The batches and stage an access group occupies.
struct Placement {
    /// Per queue: (first batch, last batch).
    extent: [Option<(usize, usize)>; 3],
    stage: PipelineStage,
}

impl Placement {
    fn new(members: &[usize], accesses: &[AccessType], task_to_batch: &[usize], queue_of: &[usize]) -> Self {
        let mut extent: [Option<(usize, usize)>; 3] = [None; 3];
        for &m in members {
            let (batch, queue) = (task_to_batch[m], queue_of[m]);
            extent[queue] = Some(match extent[queue] {
                None => (batch, batch),
                Some((lo, hi)) => (lo.min(batch), hi.max(batch)),
            });
        }
        Self {
            extent,
            stage: stage_of(accesses),
        }
    }

    /// Queue indices the group runs on.
    fn queues(&self) -> impl Iterator<Item = usize> + '_ { (0..3).filter(move |&q| self.extent[q].is_some()) }

    fn first(&self, queue: usize) -> SliceId {
        SliceId {
            batch: self.extent[queue].expect("group does not run on this queue").0,
            queue,
        }
    }

    fn last(&self, queue: usize) -> SliceId {
        SliceId {
            batch: self.extent[queue].expect("group does not run on this queue").1,
            queue,
        }
    }

    /// Earliest slice of the group.
    fn anchor(&self) -> SliceId { self.queues().map(|q| self.first(q)).min().expect("group has members") }
}

#[derive(Clone, Copy)]
struct Side<'a> {
    at: &'a Placement,
    accesses: &'a [AccessType],
}

#[derive(Clone, Copy)]
enum When {
    InCycle,
    Steady,
    Virgin,
}

#[derive(Clone, Copy)]
enum Subject {
    Image {
        image: u32,
        discard: bool,
        is_graph_owned: bool,
    },
    Buffer,
}

#[derive(Default)]
struct Anchor {
    /// (source queue, source batch, stage)
    waits: Vec<(QueueType, usize, PipelineStage)>,
    slot_waits: Vec<(QueueType, PipelineStage)>,
    acquire: bool,
    undefined_barriers: CompiledBarriers,
    barriers: CompiledBarriers,
}

impl Anchor {
    fn is_plain(&self) -> bool { self.waits.is_empty() && self.slot_waits.is_empty() && !self.acquire && self.undefined_barriers.is_empty() && self.barriers.is_empty() }

    fn image(&mut self, when: When, barrier: CompiledImageBarrier) {
        match when {
            When::InCycle => {
                self.undefined_barriers.images.push(barrier.clone());
                self.barriers.images.push(barrier);
            }
            When::Steady => self.barriers.images.push(barrier),
            When::Virgin => self.undefined_barriers.images.push(barrier),
        }
    }

    fn global(&mut self, when: When, previous: &[AccessType], next: &[AccessType]) {
        match when {
            When::InCycle => {
                self.undefined_barriers.add_global(previous, next);
                self.barriers.add_global(previous, next);
            }
            When::Steady => self.barriers.add_global(previous, next),
            When::Virgin => self.undefined_barriers.add_global(previous, next),
        }
    }
}

impl<'g, R: 'g, D: 'g> TaskGraph<'g, R, D> {
    pub(crate) fn sync_plan(&self, timelines: &[Timeline], schedule: &Schedule, alias: &AliasSchedule) -> CompiledSync {
        let queue_of: Vec<usize> = self.entries.iter().map(|entry| qi(entry.queue)).collect();
        let placements: Vec<Vec<Placement>> = timelines.iter().map(|tl| tl.groups.iter().map(|g| Placement::new(&g.members, &g.accesses, &schedule.task_to_batch, &queue_of)).collect()).collect();

        let mut derive = Derive {
            timelines,
            alias,
            placements: &placements,
            graph_owned: timelines
                .iter()
                .map(|tl| match tl.resource {
                    Resource::Image(i) => self.images[i as usize].is_graph_owned(),
                    Resource::Buffer(i) => self.buffers[i as usize].is_graph_owned(),
                })
                .collect(),
            transient: timelines
                .iter()
                .map(|tl| match tl.resource {
                    Resource::Image(i) => self.images[i as usize].is_transient(),
                    Resource::Buffer(i) => self.buffers[i as usize].is_transient(),
                })
                .collect(),
            anchors: schedule.batches.iter().map(|_| Default::default()).collect(),
        };
        for ti in 0..timelines.len() {
            derive.transitions(ti);
            derive.entry(ti);
        }

        // Present is just a node whose group is the last of its timeline.
        let present_nodes: Vec<usize> = timelines.iter().filter_map(|tl| tl.groups.last().filter(|g| g.is_present()).map(|g| g.members[0])).collect();

        self.cut(&schedule.batches, &derive.anchors, &present_nodes)
    }

    /// Folds slices into cells, resolves wait addressing and the record order.
    fn cut(&self, batches: &[Vec<usize>], anchors: &[[Anchor; 3]], present_nodes: &[usize]) -> CompiledSync {
        // A wait addresses a whole cell, so a cell must end at a wait target.
        let targets: HashSet<(usize, usize)> = anchors.iter().flat_map(|row| row.iter().flat_map(|a| a.waits.iter().map(|&(q, batch, _)| (batch, qi(q))))).collect();

        let mut cells: [Vec<Cell>; 3] = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        let mut slice_cell = vec![[usize::MAX; 3]; batches.len()];

        for (queue, &q) in QUEUES.iter().enumerate() {
            let slices = batches.iter().enumerate().filter_map(|(b, batch)| {
                let nodes: Vec<usize> = batch.iter().copied().filter(|&n| self.entries[n].queue == q).collect();
                (!nodes.is_empty()).then_some((b, nodes))
            });

            let mut open: Option<Cell> = None;
            let mut last_batch: Option<usize> = None;
            for (b, nodes) in slices {
                let anchor = &anchors[b][queue];
                let is_present_slice = nodes.iter().any(|n| present_nodes.contains(n));

                let joins = open.is_some() && anchor.is_plain() && !is_present_slice && !last_batch.is_some_and(|lb| targets.contains(&(lb, queue)));
                if !joins {
                    if let Some(done) = open.take() {
                        cells[queue].push(done);
                    }
                    open = Some(Cell {
                        queue: q,
                        wave: b,
                        nodes: Vec::new(),
                        waits: Vec::new(),
                        slot_waits: Vec::new(),
                        acquire: false,
                        terminal: false,
                        undefined_barriers: CompiledBarriers::default(),
                        barriers: CompiledBarriers::default(),
                    });
                }
                let cell = open.as_mut().expect("cell just opened");
                slice_cell[b][queue] = cells[queue].len();
                last_batch = Some(b);

                for &(src_q, src_batch, stage) in &anchor.waits {
                    push_exact(&mut cell.waits, src_q, src_batch, stage);
                }
                for &(src_q, stage) in &anchor.slot_waits {
                    push_slot(&mut cell.slot_waits, src_q, stage);
                }
                cell.acquire |= anchor.acquire;
                cell.nodes.extend(nodes);
                cell.undefined_barriers.absorb(&anchor.undefined_barriers);
                cell.barriers.absorb(&anchor.barriers);

                if is_present_slice {
                    cell.terminal = true;
                    cells[queue].push(open.take().expect("cell just opened"));
                }
            }
            if let Some(cell) = open.take() {
                cells[queue].push(cell);
            }
        }

        // Resolve source batches to their queue's cell index.
        for column in cells.iter_mut() {
            for cell in column.iter_mut() {
                let mut resolved: Vec<(QueueType, usize, PipelineStage)> = Vec::new();
                for &(src_q, src_batch, stage) in &cell.waits {
                    let src_cell = slice_cell[src_batch][qi(src_q)];
                    debug_assert!(src_cell != usize::MAX, "wait references a batch with no slice for that queue");
                    push_exact(&mut resolved, src_q, src_cell, stage);
                }
                cell.waits = resolved;
            }
        }

        let mut order = cells.iter().enumerate().flat_map(|(queue, column)| column.iter().enumerate().map(move |(k, cell)| (cell.wave, queue, k))).collect::<Vec<_>>();
        order.sort_unstable();
        let order = order.into_iter().map(|(_, queue, k)| (QUEUES[queue], k)).collect();

        CompiledSync {
            cells,
            order,
        }
    }
}

struct Derive<'a> {
    timelines: &'a [Timeline],
    alias: &'a AliasSchedule,
    /// `[timeline][group]`
    placements: &'a [Vec<Placement>],
    graph_owned: Vec<bool>,
    transient: Vec<bool>,
    /// `[batch][queue]`
    anchors: Vec<[Anchor; 3]>,
}

impl<'a> Derive<'a> {
    fn side(&self, timeline: usize, group: usize) -> Side<'a> {
        let (timelines, placements) = (self.timelines, self.placements);
        Side {
            at: &placements[timeline][group],
            accesses: &timelines[timeline].groups[group].accesses,
        }
    }

    fn anchor(&mut self, id: SliceId) -> &mut Anchor { &mut self.anchors[id.batch][id.queue] }

    /// Waits so `dst` starts after `src` finishes on every other queue.
    fn order(&mut self, src: Side, dst: Side, cycle: Cycle) {
        for dq in dst.at.queues() {
            for sq in src.at.queues().filter(|&sq| sq != dq) {
                let (dst_id, src_id) = (dst.at.first(dq), src.at.last(sq));
                match cycle {
                    Cycle::Same => {
                        debug_assert!(src_id < dst_id, "a wait must point backwards in submission order: {src_id:?} -> {dst_id:?}");
                        self.anchor(dst_id).waits.push((QUEUES[sq], src_id.batch, dst.at.stage));
                    }
                    Cycle::Previous => self.anchor(dst_id).slot_waits.push((QUEUES[sq], dst.at.stage)),
                }
            }
        }
    }

    /// The barrier that makes `dst` safe after `src`.
    fn protect(&mut self, src: Side, dst: Side, subject: Subject, when: When) {
        match subject {
            Subject::Image {
                image,
                discard,
                is_graph_owned,
            } => {
                let host = dst.at.anchor();
                self.anchor(host).image(
                    when,
                    CompiledImageBarrier {
                        image,
                        previous: src.accesses.to_vec(),
                        next: dst.accesses.to_vec(),
                        discard,
                        is_graph_owned,
                    },
                );
                // The transition runs on the host queue; other queues wait for it.
                for dq in dst.at.queues().filter(|&dq| dq != host.queue) {
                    let dst_id = dst.at.first(dq);
                    debug_assert!(host < dst_id);
                    self.anchor(dst_id).waits.push((QUEUES[host.queue], host.batch, dst.at.stage));
                }
            }
            Subject::Buffer => {
                for dq in dst.at.queues() {
                    let id = dst.at.first(dq);
                    self.anchor(id).global(when, src.accesses, dst.accesses);
                }
            }
        }
    }

    /// Consecutive conflicting groups of one timeline.
    fn transitions(&mut self, ti: usize) {
        let timelines = self.timelines;
        let tl = &timelines[ti];
        for gi in 1..tl.groups.len() {
            let (src, dst) = (self.side(ti, gi - 1), self.side(ti, gi));
            if !conflicts(src.accesses, dst.accesses) {
                continue;
            }
            self.order(src, dst, Cycle::Same);
            let subject = subject_of(&tl.resource, false, false);
            self.protect(src, dst, subject, When::InCycle);
        }
    }

    /// A timeline's first group after the previous user of its memory.
    fn entry(&mut self, ti: usize) {
        let timelines = self.timelines;
        let tl = &timelines[ti];
        let last_gi = tl.groups.len() - 1;
        let dst = self.side(ti, 0);

        // The swapchain image is acquired before its first use.
        if tl.groups[last_gi].is_present() {
            self.anchor(dst.at.anchor()).acquire = true;
        }

        let pred = self.alias.predecessor(ti);
        let src = self.side(pred.timeline, timelines[pred.timeline].groups.len() - 1);

        // A timeline that follows itself only needs sync on a real conflict.
        if pred.timeline == ti && !conflicts(src.accesses, dst.accesses) {
            return;
        }

        let owned = self.graph_owned[ti];
        // Presented images and transients start with undefined contents.
        let discard = is_write(dst.accesses) && (self.transient[ti] || src.accesses.contains(&AccessType::Present));

        match pred.cycle {
            Cycle::Same => {
                self.order(src, dst, Cycle::Same);
                self.protect(src, dst, subject_of(&tl.resource, discard, false), When::InCycle);
            }
            Cycle::Previous => {
                if owned {
                    self.order(src, dst, Cycle::Previous);
                }
                self.protect(src, dst, subject_of(&tl.resource, discard, owned), When::Steady);

                if let (Resource::Image(image), true) = (&tl.resource, is_write(dst.accesses)) {
                    let nothing = Side {
                        at: dst.at,
                        accesses: &[AccessType::None],
                    };
                    let subject = Subject::Image {
                        image: *image,
                        discard: true,
                        is_graph_owned: false,
                    };
                    self.protect(nothing, dst, subject, When::Virgin);
                }
            }
        }
    }
}

fn subject_of(resource: &Resource, discard: bool, is_graph_owned: bool) -> Subject {
    match resource {
        Resource::Image(image) => Subject::Image {
            image: *image,
            discard,
            is_graph_owned,
        },
        Resource::Buffer(_) => Subject::Buffer,
    }
}

fn push_exact(waits: &mut Vec<(QueueType, usize, PipelineStage)>, src_q: QueueType, src: usize, stage: PipelineStage) {
    match waits.iter_mut().find(|(q, b, _)| *q == src_q && *b == src) {
        Some(existing) => existing.2 |= stage,
        None => waits.push((src_q, src, stage)),
    }
}

fn push_slot(waits: &mut Vec<(QueueType, PipelineStage)>, src_q: QueueType, stage: PipelineStage) {
    match waits.iter_mut().find(|(q, _)| *q == src_q) {
        Some(existing) => existing.1 |= stage,
        None => waits.push((src_q, stage)),
    }
}

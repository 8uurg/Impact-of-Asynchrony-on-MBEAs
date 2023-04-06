use std::{cell::Cell, cmp::Ordering, collections::HashMap};

struct Element {
    v: f64,
    k: usize,
}

/// Define the ordering we'll be using among the elements.

fn cmp_f64(s: &f64, o: &f64) -> Ordering {
    match (s.is_nan(), o.is_nan(), s, o) {
        (true, true, _, _) => Ordering::Equal,
        (true, false, _, _) => Ordering::Less,
        (false, true, _, _) => Ordering::Greater,
        (false, false, x, y) if x > y => Ordering::Greater,
        (false, false, x, y) if y > x => Ordering::Less,
        _ => Ordering::Equal,
    }
}

/// An index into the dual-heap.
///
///  0 - root, or centroid of both of the heaps.
/// -1 - root of left heap
///  1 - root of right heap
///
/// All elements >= 1 are part of the right heap.
/// While all elements <= -1 are part of the left heap.
///
/// Furthermore, the heap preserves the property that
///     v(H(-1)) <= v(H(0)) <= v(H(1))
///
/// Whereas each subheap guarantees
///     v(H(x)) where x < -1 <= v(H(-1))
///     v(H(x)) where x > 1 >= v(H(1))
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct HeapIndex(isize);

pub enum QuantileKind {
    LinearInterp,
    Nearest,
    LazyMiddle,
}

/// A Dual-Heap for incrementally computing updating quantile `q`.
///
/// Inspired & Implemented according to:
///     Hardle, W., and W. Steiger. 1995.
///     ‘Algorithm AS 296: Optimal Median Smoothing’.
///     Applied Statistics 44 (2): 258.
///     https://doi.org/10.2307/2986349.
/// and
///     https://aakinshin.net/posts/partitioning-heaps-quantile-estimator/
///
/// It ensures that it keeps track of the three items with rank [|_ q * |H| _| - 1, |_ q * |H| _|, |_ q * |H| _| + 1]
/// i.e. the solutions bracketing the desired quantile.
///
/// It does so by preserving the property that
///     v(lh(0)) <= e <= v(rh(1))
///
/// Combining this with that fact that each subheap guarantees respectively:
///     v(lh(x)) <= v(lh(-1)) -- lh is a maxheap
///     v(rh(x)) >= v(rh( 1)) -- rh is a minheap
///
/// Leads to the fact that lh(0), e, rh(0) have rank |lh|, |lh| + 1, |lh| + 2 respectively.
///
/// By ensuring that |lh| = |_ q * |H| _| - 1, we can keep track of the right three items necessary
/// to compute the given quantile - with interpolation, if necessary.
pub struct DualHeap {
    e: Option<Element>,
    lh: Vec<Element>,
    rh: Vec<Element>,
    ktohi: HashMap<usize, Cell<HeapIndex>>,
    q: f64,
}

fn parent_idx(idx: usize) -> usize {
    idx >> 1
}
fn child_a_idx(idx: usize) -> usize {
    (idx << 1) + 1
}

impl DualHeap {
    pub fn new(q: f64) -> DualHeap {
        DualHeap {
            e: None,
            lh: vec![],
            rh: vec![],
            ktohi: HashMap::<usize, Cell<HeapIndex>>::new(),
            q: q,
        }
    }

    fn lh_swap(&mut self, a: usize, b: usize) {
        self.lh.swap(a, b);
        self.ktohi[&self.lh[a].k].swap(&self.ktohi[&self.lh[b].k]);
    }
    fn rh_swap(&mut self, a: usize, b: usize) {
        self.rh.swap(a, b);
        self.ktohi[&self.rh[a].k].swap(&self.ktohi[&self.rh[b].k]);
    }

    fn heapify_up_lh(&mut self, idx: usize) {
        let mut c = idx;
        // While we are not the root, and we are greater than the parent (lh is MaxHeap).
        while c > 0 && cmp_f64(&self.lh[c].v, &self.lh[parent_idx(c)].v) == Ordering::Greater {
            // Swap with the parent
            self.lh_swap(c, parent_idx(c));
            c = parent_idx(c);
        }
    }
    fn heapify_down_lh(&mut self, idx: usize) {
        let mut c = idx;
        let mut loopcount = 0;
        loop {
            let cv = &self.lh[c];
            // Find children
            let ca = self.lh.get(child_a_idx(c));
            let cb = self.lh.get(child_a_idx(c) + 1);
            match (ca, cb) {
                (None, None) => return,            // We are at a leaf, we can stop.
                (None, Some(_)) => unreachable!(), // Unreachable: Only earlier indices can be missing.
                (Some(e), None) if cmp_f64(&e.v, &cv.v) == Ordering::Greater => {
                    // Swap the child and parent
                    self.lh_swap(c, child_a_idx(c));
                    // There is only one way to have one child, and that means we are now a leaf.
                    return;
                }
                (Some(_), None) => return, // Only child is smaller, child is leaf, we are done.
                (Some(a), Some(b)) => {
                    // If greater of the two invalidates current, then greater of the two should replace current.
                    match cmp_f64(&a.v, &b.v) {
                        Ordering::Greater | Ordering::Equal => {
                            // a is greater!
                            self.lh_swap(c, child_a_idx(c));
                            c = child_a_idx(c);
                        }
                        Ordering::Less => {
                            // b is greater
                            self.lh_swap(c, child_a_idx(c) + 1);
                            c = child_a_idx(c) + 1;
                        }
                    }
                }
            };
            loopcount += 1;
            if loopcount > self.lh.len() {
                panic!("Something is wrong!");
            }
        };
    }
    fn heapify_lh(&mut self, idx: usize) {
        // Can we heapify upwards?
        if idx != 0 && cmp_f64(&self.lh[idx].v, &self.lh[idx >> 1].v) == Ordering::Greater {
            return self.heapify_up_lh(idx);
        }
        // Otherwise: heapify downwards
        self.heapify_down_lh(idx);
    }
    fn heapify_up_rh(&mut self, idx: usize) {
        let mut c = idx;
        // While we are not the root, and we are less than the parent (rh is MinHeap).
        while c > 0 && cmp_f64(&self.rh[c].v, &self.rh[parent_idx(c)].v) == Ordering::Less {
            // Swap with the parent
            self.rh_swap(c, parent_idx(c));
            c = parent_idx(c);
        }
    }
    fn heapify_down_rh(&mut self, idx: usize) {
        let mut c = idx;
        let mut loopcount = 0;
        loop {
            let cv = &self.rh[c];
            // Find children
            let ca = self.rh.get(child_a_idx(c));
            let cb = self.rh.get(child_a_idx(c) + 1);
            match (ca, cb) {
                (None, None) => return,            // We are at a leaf, we can stop.
                (None, Some(_)) => unreachable!(), // Unreachable: Only earlier indices can be missing.
                (Some(e), None) if cmp_f64(&e.v, &cv.v) == Ordering::Less => {
                    // Swap the child and parent
                    self.rh_swap(c, child_a_idx(c));
                    // There is only one way to have one child, and that means we are now a leaf.
                    return;
                }
                (Some(_), None) => return, // Only child is smaller, child is leaf, we are done.
                (Some(a), Some(b)) => {
                    // If lesser of the two invalidates current, then lesser of the two should replace current.
                    match cmp_f64(&a.v, &b.v) {
                        Ordering::Less | Ordering::Equal => {
                            // a is less!
                            self.rh_swap(c, child_a_idx(c));
                            c = child_a_idx(c);
                        }
                        Ordering::Greater => {
                            // b is less!
                            self.rh_swap(c, child_a_idx(c) + 1);
                            c = child_a_idx(c) + 1;
                        }
                    }
                }
            };
            loopcount += 1;
            if loopcount > self.rh.len() {
                panic!("Something is wrong!");
            }
        };
    }
    fn heapify_rh(&mut self, idx: usize) {
        // Can we heapify upwards?
        if idx != 0 && cmp_f64(&self.rh[idx].v, &self.rh[idx >> 1].v) == Ordering::Less {
            return self.heapify_up_rh(idx);
        }
        // Otherwise: heapify downwards
        self.heapify_down_rh(idx);
    }
    fn insert_lh(&mut self, e: Element) {
        let k = e.k;
        self.lh.push(e);
        self.ktohi
            .insert(k, Cell::new(HeapIndex(-(self.lh.len() as isize))));
        self.heapify_up_lh(self.lh.len() - 1);
    }
    fn insert_rh(&mut self, e: Element) {
        let k = e.k;
        self.rh.push(e);
        self.ktohi
            .insert(k, Cell::new(HeapIndex((self.rh.len() as isize))));
        self.heapify_up_rh(self.rh.len() - 1);
    }
    fn pop_root_lh(&mut self) -> Element {
        // Swap the last element in the heap to the root.
        self.lh_swap(0, self.lh.len() - 1);
        // Pop what was originally the root.
        let orig_root = self.lh.pop().unwrap();
        // New root needs to be heapified downwards, unless: there is no root...
        if self.lh.len() > 0 {
            self.heapify_down_lh(0);
        }
        self.ktohi.remove(&orig_root.k);
        orig_root
    }
    fn pop_root_rh(&mut self) -> Element {
        // Swap the last element in the heap to the idx.
        self.rh_swap(0, self.rh.len() - 1);
        // Pop what was originally the root.
        let orig_root = self.rh.pop().unwrap();
        // New root needs to be heapified downwards, unless: there is no root...
        if self.rh.len() > 0 {
            self.heapify_down_rh(0);
        }
        self.ktohi.remove(&orig_root.k);
        orig_root
    }
    fn remove_idx_lh(&mut self, idx: usize) {
        // Swap the last element in the heap to the idx.
        self.lh_swap(idx, self.lh.len() - 1);
        // Remove original item
        let orig_root = self.lh.pop().unwrap();
        self.ktohi.remove(&orig_root.k);
        // Heapify replacement, unless this is beyond the end of the heap
        if self.lh.len() > idx {
            self.heapify_lh(idx);
        }
    }
    fn remove_idx_rh(&mut self, idx: usize) {
        // Swap the last element in the heap to the right position.
        self.rh_swap(idx, self.rh.len() - 1);
        // Remove original item
        let orig_root = self.rh.pop().unwrap();
        self.ktohi.remove(&orig_root.k);
        // Heapify replacement, unless this is beyond the end of the heap
        if self.rh.len() > idx {
            self.heapify_rh(idx);
        }
    }

    /// e -> rh, root lh -> e
    /// Only call this function if e is Some.
    fn rotate_right(&mut self) {
        let e_orig = self.e.take().unwrap();
        self.insert_rh(e_orig);
        let new_root = self.pop_root_lh();
        let new_root_k = new_root.k;
        self.e = Some(new_root);

        self.ktohi.insert(new_root_k, Cell::new(HeapIndex(0)));
    }
    // e -> lh, root rh -> e
    /// Only call this function if e is Some.
    fn rotate_left(&mut self) {
        let e_orig = self.e.take().unwrap();
        self.insert_lh(e_orig);

        let new_root = self.pop_root_rh();
        let new_root_k = new_root.k;
        self.e = Some(new_root);

        self.ktohi.insert(new_root_k, Cell::new(HeapIndex(0)));
    }

    pub fn add(&mut self, k: usize, v: f64) {
        // If item is already within the heap, ignore (or maybe: panic)
        if self.ktohi.contains_key(&k) {
            return;
        }

        // If root is empty, place item at root.
        if let None = self.e {
            self.e = Some(Element { v, k });
            self.ktohi.insert(k, Cell::new(HeapIndex(0)));
            return;
        }

        // Compute preferred sizes after addition
        let l_after = 2 + self.lh.len() + self.rh.len();
        let lh_p = f64::floor((l_after as f64) * self.q);
        let rh_p = f64::floor((l_after as f64) * (1.0 - self.q));

        // Depending on the comparison against e, the item should either end up in the min,
        // or maxheap.
        // Unwrap is OK as "if let None = self.e" a few lines prior ensures that self.e is Some.
        match (
            cmp_f64(&v, &self.e.as_ref().unwrap().v),
            (self.lh.len() as f64) < lh_p,
            (self.rh.len() as f64) < rh_p,
        ) {
            (Ordering::Less, _, _) => self.insert_lh(Element { v, k }), // Must insert into left heap.
            (Ordering::Equal, true, _) => self.insert_lh(Element { v, k }), // Add to heap not at target capacity
            (Ordering::Equal, _, true) => self.insert_rh(Element { v, k }), // Add to heap not at target capacity
            (Ordering::Greater, _, _) => self.insert_rh(Element { v, k }), // Must insert into right heap.
            (Ordering::Equal, false, false) => unreachable!(), // Should not be reachable: one of the two must have space for insertion.
        }

        // Rebalance if necessary.
        self.rebalance(lh_p, rh_p);
    }

    fn rebalance(&mut self, lh_p: f64, rh_p: f64) {
        match (
            (self.lh.len() as f64) <= lh_p,
            (self.rh.len() as f64) <= rh_p,
        ) {
            (true, true) => {}                    // Heaps are fine!
            (true, false) => self.rotate_left(), // Right Heap is too large, rotate elements towards left.
            (false, true) => self.rotate_right(), // Left heap is too large, rotate elements towards right.
            (false, false) => unreachable!(), // We have only changed one of the two heaps, so this should never happen.
        }
    }

    pub fn remove(&mut self, k: usize) {
        // Get location of k in the heap.
        let loc = match self.ktohi.get_mut(&k) {
            // If item is not within the heap, ignore request for removal.
            None => return,
            Some(e) => e,
        };

        // Edge case, if heaps are empty, the only item that can still be removed is e.
        // Given that the map only contains elements contained within the list, it must be e that is being removed.
        // As this case is a bit annoying: it wraps-around the preferred sizes for the heaps(!)
        // So take the shortcut here, and just remove the root & clear the key lookup table:
        if self.lh.len() == 0 && self.rh.len() == 0 {
            self.e.take();
            self.ktohi.clear();
            return;
        }

        // Compute preferred sizes after removal
        let l_after = self.lh.len() + self.rh.len() - 1; // Does not wrap, as self.lh.len() + self.rh.len() >= 1.
        let lh_p = f64::floor((l_after as f64) * self.q);
        let rh_p = f64::floor((l_after as f64) * (1.0 - self.q));

        // Remove item
        match (
            loc.get_mut(),
            (self.lh.len() as f64) < lh_p,
            (self.rh.len() as f64) < rh_p,
            self.lh.is_empty(),
            self.rh.is_empty(),
        ) {
            (&mut HeapIndex(i), _, _, _, _) if i < 0 => {
                let kv: usize = (-i - 1).try_into().unwrap();
                self.remove_idx_lh(kv)
            }
            (&mut HeapIndex(0), _, false, _, false) => {
                let orig_root = self.pop_root_rh();
                let orig_root_k = orig_root.k;
                self.ktohi.remove(&self.e.as_ref().unwrap().k);
                self.e.replace(orig_root); // Remove right root to free up space.
                self.ktohi.insert(orig_root_k, Cell::new(HeapIndex(0)));
            }
            (&mut HeapIndex(0), false, _, false, _) => {
                let orig_root = self.pop_root_lh();
                let orig_root_k = orig_root.k;
                self.ktohi.remove(&self.e.as_ref().unwrap().k);
                self.e.replace(orig_root); // Remove left root to free up space.
                self.ktohi.insert(orig_root_k, Cell::new(HeapIndex(0)));
            }
            (&mut HeapIndex(i), _, _, _, _) if i > 0 => {
                let kv: usize = (i - 1).try_into().unwrap();
                self.remove_idx_rh(kv)
            }
            (_, _, _, _, _) => unreachable!(),
        }

        // Rebalance, if necessary.
        self.rebalance(lh_p, rh_p);
    }

    pub fn quantile(&self, qk: &QuantileKind) -> Option<f64> {
        // If e is none, the result should be none as well -> heap is empty.
        let e = self.e.as_ref()?;
        // Compute the ranks of each element.
        let rank_e = self.lh.len() as isize;
        let rank_root_lh = rank_e - 1;
        let rank_root_rh = rank_e + 1;
        // Compute current size of the heap
        let len = self.lh.len() + self.rh.len();
        // Shortcut: single sample, all quantiles are equal to this single sample, no matter what.
        if len == 0 {
            return Some(e.v);
        }

        let rank_value_wanted = (len as f64) * self.q;

        match qk {
            QuantileKind::LinearInterp => {
                // Close enough? Return the actual value.
                if (rank_value_wanted - (rank_e as f64)).abs() < f64::EPSILON {
                    return Some(e.v);
                }
                // Which one is on the left, and which one on the right?
                let (left, left_v, right, right_v) = if (rank_e as f64) > rank_value_wanted {
                    (
                        rank_root_lh as f64,
                        self.lh.first().unwrap().v,
                        rank_e as f64,
                        e.v,
                    )
                } else {
                    (
                        rank_e as f64,
                        e.v,
                        rank_root_rh as f64,
                        self.rh.first().unwrap().v,
                    )
                };
                //
                Some(left_v + (rank_value_wanted - left) / (right - left) * (right_v - left_v))
            }
            QuantileKind::Nearest => {
                let delta_e = (rank_value_wanted - (rank_e as f64)).abs();
                let delta_lh = (rank_value_wanted - (rank_root_lh as f64)).abs();
                let delta_rh = (rank_value_wanted - (rank_root_rh as f64)).abs();
                if delta_e < delta_lh {
                    if delta_e < delta_rh {
                        Some(e.v)
                    } else {
                        Some(self.rh.first().unwrap().v)
                    }
                } else {
                    if delta_lh < delta_rh {
                        Some(self.lh.first().unwrap().v)
                    } else {
                        Some(self.rh.first().unwrap().v)
                    }
                }
            }
            QuantileKind::LazyMiddle => Some(e.v),
        }
    }

    pub fn len(&self) -> usize {
        usize::from(self.e.is_some()) + self.lh.len() + self.rh.len()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn assert_dh_valid_state(dh: &DualHeap)
    {
        // Get center -- if some, we can do other checks.
        let e= match &dh.e {
            Some(e) => e,
            None => {
                // if None, the heap should be entirely empty.
                assert_eq!(dh.lh.len(), 0);
                assert_eq!(dh.rh.len(), 0);
                assert_eq!(dh.ktohi.len(), 0);
                return;
            }
        };
        // Ensure index is correctly stored.
        assert_eq!(dh.ktohi.get(&e.k).unwrap().get(), HeapIndex(0));
        // validate property around center: lh_root.v < e.v < rh_root.v
        let lt = dh.lh.first();
        if let Some(v) = lt {
            // check correct index
            assert_eq!(dh.ktohi.get(&v.k).unwrap().get(), HeapIndex(-1));
            // ensure order
            assert!(v.v < e.v);
        }
        let rh = dh.rh.first();
        if let Some(v) = rh {
            // check correct index
            assert_eq!(dh.ktohi.get(&v.k).unwrap().get(), HeapIndex(1));
            // ensure order
            assert!(e.v < v.v);
        }
        // lh is a maxheap & ensure correct indexing
        for idx in 1..dh.lh.len() {
            let idx_is: isize = idx.try_into().unwrap();
            let v = &dh.lh[idx];
            // check correct index
            assert_eq!(dh.ktohi.get(&v.k).unwrap().get(), HeapIndex(-idx_is - 1));
            // check ordering
            let parent = &dh.lh[parent_idx(idx)];
            assert!(parent.v > v.v);
        }

        // rh is a minheap
        for idx in 1..dh.rh.len() {
            let idx_is: isize = idx.try_into().unwrap();
            let v = &dh.rh[idx];
            // check correct index
            assert_eq!(dh.ktohi.get(&v.k).unwrap().get(), HeapIndex(idx_is + 1));
            // check ordering
            let parent = &dh.rh[parent_idx(idx)];
            assert!(parent.v < v.v);
        }
    }

    #[test]
    fn test_simple_add() {
        let mut dh = DualHeap::new(0.5);
        dh.add(0, 0.0);
        assert_dh_valid_state(&dh);
    }

    fn create_test_heap() -> DualHeap {
        let mut dh = DualHeap::new(0.5);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), None);
        dh.add(0, 0.0); //
        dh.add(1, 1.0); //
        dh.add(2, 2.0); //
        dh.add(3, 3.0); // Root left
        dh.add(4, 4.0); // Median
        dh.add(5, 5.0); // Root right
        dh.add(6, 6.0); // 
        dh.add(7, 7.0); //
        dh.add(8, 8.0); //
        dh
    }

    #[test]
    fn test_a_removal_0() {
        let mut t = create_test_heap();

        assert_dh_valid_state(&t);
        t.remove(0);
        assert_dh_valid_state(&t);
    }
    #[test]
    fn test_a_removal_1() {
        let mut t = create_test_heap();
        
        assert_dh_valid_state(&t);
        t.remove(1);
        assert_dh_valid_state(&t);
    }
    #[test]
    fn test_a_removal_2() {
        let mut t = create_test_heap();
        
        assert_dh_valid_state(&t);
        t.remove(2);
        assert_dh_valid_state(&t);
    }
    #[test]
    fn test_a_removal_3() {
        let mut t = create_test_heap();
        
        assert_dh_valid_state(&t);
        t.remove(4);
        assert_dh_valid_state(&t);
    }
    #[test]
    fn test_a_removal_4() {
        let mut t = create_test_heap();
        
        assert_dh_valid_state(&t);
        t.remove(4);
        assert_dh_valid_state(&t);
    }
    #[test]
    fn test_a_removal_5() {
        let mut t = create_test_heap();
        
        assert_dh_valid_state(&t);
        t.remove(5);
        assert_dh_valid_state(&t);
    }
    #[test]
    fn test_a_removal_6() {
        let mut t = create_test_heap();
        
        assert_dh_valid_state(&t);
        t.remove(6);
        assert_dh_valid_state(&t);
    }
    #[test]
    fn test_a_removal_7() {
        let mut t = create_test_heap();
        
        assert_dh_valid_state(&t);
        t.remove(7);
        assert_dh_valid_state(&t);
    }
    #[test]
    fn test_a_removal_8() {
        let mut t = create_test_heap();
        
        assert_dh_valid_state(&t);
        t.remove(8);
        assert_dh_valid_state(&t);
    }

    #[test]
    fn test_multiple_add() {
        let mut dh = DualHeap::new(0.5);
        dh.add(0, 0.0);
        assert_dh_valid_state(&dh);
        dh.add(1, 1.0);
        assert_dh_valid_state(&dh);
        dh.add(2, 2.0);
        assert_dh_valid_state(&dh);
    }

    #[test]
    fn test_simple_add_remove() {
        let mut dh = DualHeap::new(0.5);
        dh.add(0, 0.0);
        assert_dh_valid_state(&dh);
        dh.remove(0);
        assert_dh_valid_state(&dh);
    }
    #[test]
    fn test_multiple_add_remove() {
        let mut dh = DualHeap::new(0.5);
        dh.add(0, 0.0);
        assert_dh_valid_state(&dh);
        dh.add(1, 1.0);
        assert_dh_valid_state(&dh);
        dh.add(2, 2.0);
        assert_dh_valid_state(&dh);
        dh.remove(2);
        assert_dh_valid_state(&dh);
        dh.remove(1);
        assert_dh_valid_state(&dh);
        dh.remove(0);
        assert_dh_valid_state(&dh);
    }

    #[test]
    fn test_median_increasing_sequence_linear_interp() {
        let mut dh = DualHeap::new(0.5);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), None);
        dh.add(0, 0.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(0.0));
        dh.add(1, 1.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(0.5));
        dh.add(2, 2.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(1.0));
        dh.add(3, 3.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(1.5));
        dh.remove(0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(2.0));
        dh.remove(1);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(2.5));
        dh.remove(2);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(3.0));
        dh.remove(3);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), None);
    }
    #[test]
    fn test_min_increasing_sequence_linear_interp() {
        let mut dh = DualHeap::new(0.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), None);
        dh.add(0, 0.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(0.0));
        dh.add(1, 1.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(0.0));
        dh.add(2, 2.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(0.0));
        dh.add(3, 3.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(0.0));
        dh.remove(0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(1.0));
        dh.remove(1);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(2.0));
        dh.remove(2);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(3.0));
        dh.remove(3);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), None);
    }
    #[test]
    fn test_max_increasing_sequence_linear_interp() {
        let mut dh = DualHeap::new(1.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), None);
        dh.add(0, 0.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(0.0));
        dh.add(1, 1.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(1.0));
        dh.add(2, 2.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(2.0));
        dh.add(3, 3.0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(3.0));
        dh.remove(0);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(3.0));
        dh.remove(1);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(3.0));
        dh.remove(2);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), Some(3.0));
        dh.remove(3);
        assert_eq!(dh.quantile(&QuantileKind::LinearInterp), None);
    }

    #[test]
    fn test_median_increasing_sequence_nearest() {
        let mut dh = DualHeap::new(0.5);
        assert_eq!(dh.quantile(&QuantileKind::Nearest), None);
        dh.add(0, 0.0);
        assert_eq!(dh.quantile(&QuantileKind::Nearest), Some(0.0));
        dh.add(1, 1.0);
        dh.add(2, 2.0);
        assert_eq!(dh.quantile(&QuantileKind::Nearest), Some(1.0));
        dh.add(3, 3.0);
        dh.remove(0);
        assert_eq!(dh.quantile(&QuantileKind::Nearest), Some(2.0));
        dh.remove(1);
        dh.remove(2);
        assert_eq!(dh.quantile(&QuantileKind::Nearest), Some(3.0));
        dh.remove(3);
        assert_eq!(dh.quantile(&QuantileKind::Nearest), None);
    }

    #[test]
    fn test_median_increasing_sequence_lazymiddle() {
        let mut dh = DualHeap::new(0.5);
        assert_eq!(dh.quantile(&QuantileKind::LazyMiddle), None);
        dh.add(0, 0.0);
        assert_eq!(dh.quantile(&QuantileKind::LazyMiddle), Some(0.0));
        dh.add(1, 1.0);
        dh.add(2, 2.0);
        assert_eq!(dh.quantile(&QuantileKind::LazyMiddle), Some(1.0));
        dh.add(3, 3.0);
        dh.remove(0);
        assert_eq!(dh.quantile(&QuantileKind::LazyMiddle), Some(2.0));
        dh.remove(1);
        dh.remove(2);
        assert_eq!(dh.quantile(&QuantileKind::LazyMiddle), Some(3.0));
        dh.remove(3);
        assert_eq!(dh.quantile(&QuantileKind::LazyMiddle), None);
    }

}

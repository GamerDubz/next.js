//! A set of `Copy + Ord` values stored as sorted arrays, for sets that are built once and then
//! mostly read, like a task's forward dependency edges.
//!
//! A finished set stores exactly its elements: no empty buckets, control bytes or power-of-two
//! rounding as in a hash table. Lookups are binary searches, which for the small integer keys these
//! sets hold are about as fast as hashing up to thousands of elements, and iteration is in order.
//!
//! # Representation
//! * Up to `INLINE` elements are stored inline, sorted.
//! * Past that, the elements live in a heap allocation behind a thin pointer, after a small header
//!   holding the capacity and the length of the *main* run. The elements are the sorted main run
//!   followed by a sorted *tail* of recent insertions. Inserting into the middle of a large array
//!   costs O(n), so insertions go into the (short) tail instead, which is merged into the main run
//!   once it outgrows about `sqrt(len)`. Inserting in ascending order, the common case for freshly
//!   allocated ids, extends the main run directly. Lookups search both runs.
//! * [`SortedSet::shrink_to_fit`] merges the tail and drops spare capacity. A set shrunk to at most
//!   `INLINE` elements moves back inline.
//!
//! The encoded form (the length, then the elements) is the same as an [`AutoSet`]'s.
//!
//! [`AutoSet`]: crate::AutoSet

use std::{
    alloc::{self, Layout},
    fmt::{self, Debug},
    mem::{ManuallyDrop, MaybeUninit},
    ptr::{self, NonNull},
    slice,
};

use bincode::{
    Decode, Encode,
    de::Decoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
    impl_borrow_decode,
};
use shrink_to_fit::ShrinkToFit;

/// Set in [`SortedSet::len`] while the elements live on the heap.
const SPILLED: u32 = 1 << 31;
/// The largest number of elements a set can hold.
const MAX_LEN: usize = (SPILLED - 1) as usize;
/// The tail is merged into the main run once longer than this or `sqrt(len)`.
const MIN_TAIL: usize = 16;

pub struct SortedSet<T, const INLINE: usize = 0> {
    /// The number of elements, with [`SPILLED`] set while they live on the heap.
    len: u32,
    data: Data<T, INLINE>,
}

union Data<T, const INLINE: usize> {
    inline: ManuallyDrop<[MaybeUninit<T>; INLINE]>,
    /// Valid while spilled: a [`Header`] followed by `cap` element slots.
    heap: NonNull<Header>,
}

/// The start of a spilled set's allocation.
#[repr(C)]
struct Header {
    cap: u32,
    /// The length of the sorted main run. The elements after it, up to `len`, are the sorted tail.
    main: u32,
}

// SAFETY: a `SortedSet` owns its elements like a `Vec` does.
unsafe impl<T: Send, const INLINE: usize> Send for SortedSet<T, INLINE> {}
// SAFETY: shared access only reads the elements.
unsafe impl<T: Sync, const INLINE: usize> Sync for SortedSet<T, INLINE> {}

impl<T, const INLINE: usize> Default for SortedSet<T, INLINE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const INLINE: usize> SortedSet<T, INLINE> {
    pub const fn new() -> Self {
        Self {
            len: 0,
            data: Data {
                inline: ManuallyDrop::new([const { MaybeUninit::uninit() }; INLINE]),
            },
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        (self.len & !SPILLED) as usize
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    const fn is_spilled(&self) -> bool {
        self.len & SPILLED != 0
    }

    /// The layout of an allocation with room for `cap` elements, and the offset of the elements.
    fn layout(cap: usize) -> (Layout, usize) {
        let (layout, offset) = Layout::new::<Header>()
            .extend(Layout::array::<T>(cap).expect("SortedSet capacity overflow"))
            .expect("SortedSet capacity overflow");
        (layout.pad_to_align(), offset)
    }

    /// The offset of the elements in the allocation, which doesn't depend on the capacity.
    fn elements_offset() -> usize {
        Self::layout(0).1
    }

    /// # Safety
    /// Must be spilled.
    unsafe fn header(&self) -> &Header {
        // SAFETY: spilled, so `heap` points to a live allocation starting with a `Header`.
        unsafe { self.data.heap.as_ref() }
    }

    /// # Safety
    /// Must be spilled.
    unsafe fn header_mut(&mut self) -> &mut Header {
        // SAFETY: see `header`.
        unsafe { self.data.heap.as_mut() }
    }

    fn as_ptr(&self) -> *const T {
        if self.is_spilled() {
            // SAFETY: spilled, and the elements follow the header at `elements_offset`.
            unsafe {
                self.data
                    .heap
                    .as_ptr()
                    .cast::<u8>()
                    .add(Self::elements_offset())
                    .cast::<T>()
            }
        } else {
            // SAFETY: not spilled, so `inline` is the active field.
            unsafe { self.data.inline.as_ptr().cast::<T>() }
        }
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        if self.is_spilled() {
            // A heap pointer carries the allocation's provenance, so going through `as_ptr` is
            // fine.
            self.as_ptr().cast_mut()
        } else {
            // SAFETY: not spilled, so `inline` is the active field. Derived from `&mut self` so the
            // pointer may write.
            unsafe { (*self.data.inline).as_mut_ptr().cast::<T>() }
        }
    }

    /// All elements: the sorted main run followed by the sorted tail.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the first `len` elements are initialized.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// The length of the sorted main run. Inline elements are always fully sorted.
    fn main_len(&self) -> usize {
        if self.is_spilled() {
            // SAFETY: spilled.
            unsafe { self.header().main as usize }
        } else {
            self.len()
        }
    }

    /// The sorted main run and the sorted tail.
    fn runs(&self) -> (&[T], &[T]) {
        self.as_slice().split_at(self.main_len())
    }

    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// The capacity of the heap allocation, or `0` while inline.
    fn heap_capacity(&self) -> usize {
        if self.is_spilled() {
            // SAFETY: spilled.
            unsafe { self.header().cap as usize }
        } else {
            0
        }
    }

    /// Bytes of the heap allocation, `0` while inline. Excludes heap memory owned by the elements.
    pub fn heap_size(&self) -> usize {
        if self.is_spilled() {
            Self::layout(self.heap_capacity()).0.size()
        } else {
            0
        }
    }

    /// Element slots in the heap allocation, `0` while inline.
    pub fn heap_slots(&self) -> usize {
        self.heap_capacity()
    }

    /// Allocates room for `cap` elements, with an initialized header.
    fn allocate(cap: usize, main: usize) -> NonNull<Header> {
        assert!(cap <= MAX_LEN, "SortedSet capacity overflow");
        let (layout, _) = Self::layout(cap);
        // SAFETY: the layout has a non-zero size, it contains a `Header`.
        let ptr = unsafe { alloc::alloc(layout) }.cast::<Header>();
        let Some(ptr) = NonNull::new(ptr) else {
            alloc::handle_alloc_error(layout);
        };
        // SAFETY: freshly allocated with room and alignment for a `Header`.
        unsafe {
            ptr.write(Header {
                cap: cap as u32,
                main: main as u32,
            })
        };
        ptr
    }

    /// Frees the heap allocation, without touching the elements (they are `Copy`, see the impls).
    ///
    /// # Safety
    /// Must be spilled. Afterwards `data.heap` dangles until replaced.
    unsafe fn deallocate(&mut self) {
        // SAFETY: spilled, so `heap` was allocated by `allocate`/`set_capacity` with this layout.
        unsafe {
            let (layout, _) = Self::layout(self.heap_capacity());
            alloc::dealloc(self.data.heap.as_ptr().cast(), layout);
        }
    }

    /// Resizes the heap allocation to `cap` elements.
    ///
    /// # Safety
    /// Must be spilled, and `cap >= len`.
    unsafe fn set_capacity(&mut self, cap: usize) {
        debug_assert!(self.is_spilled() && cap >= self.len());
        assert!(cap <= MAX_LEN, "SortedSet capacity overflow");
        let (old_layout, _) = Self::layout(self.heap_capacity());
        let (new_layout, _) = Self::layout(cap);
        // SAFETY: allocated with `old_layout`; the new size is non-zero. The header and the first
        // `len <= cap` elements are in the preserved prefix, at the same offsets.
        unsafe {
            let ptr = alloc::realloc(
                self.data.heap.as_ptr().cast(),
                old_layout,
                new_layout.size(),
            );
            let Some(ptr) = NonNull::new(ptr.cast::<Header>()) else {
                alloc::handle_alloc_error(new_layout);
            };
            self.data.heap = ptr;
            self.header_mut().cap = cap as u32;
        }
    }

    pub fn clear(&mut self) {
        if self.is_spilled() {
            // SAFETY: spilled; `data` is replaced right after.
            unsafe { self.deallocate() };
        }
        *self = Self::new();
    }
}

impl<T: Copy + Ord, const INLINE: usize> SortedSet<T, INLINE> {
    /// Builds a set from strictly ascending elements, with exactly their capacity.
    fn from_sorted_vec(elements: Vec<T>) -> Self {
        debug_assert!(elements.is_sorted_by(|a, b| a < b));
        let mut set = Self::new();
        let len = elements.len();
        if len > INLINE {
            set.data = Data {
                heap: Self::allocate(len, len),
            };
            set.len = len as u32 | SPILLED;
        } else {
            set.len = len as u32;
        }
        // SAFETY: the destination has room for `len` elements and doesn't overlap the source.
        unsafe { ptr::copy_nonoverlapping(elements.as_ptr(), set.as_mut_ptr(), len) };
        set
    }

    pub fn contains(&self, value: &T) -> bool {
        let (main, tail) = self.runs();
        main.binary_search(value).is_ok() || tail.binary_search(value).is_ok()
    }

    /// Inserts `value`, returning whether it was not in the set yet.
    pub fn insert(&mut self, value: T) -> bool {
        let len = self.len();
        if !self.is_spilled() {
            let pos = match self.as_slice().binary_search(&value) {
                Ok(_) => return false,
                Err(pos) => pos,
            };
            if len < INLINE {
                let ptr = self.as_mut_ptr();
                // SAFETY: `len < INLINE`, so there is room to shift `pos..len` up by one.
                unsafe {
                    ptr::copy(ptr.add(pos), ptr.add(pos + 1), len - pos);
                    ptr.add(pos).write(value);
                }
                self.len += 1;
                return true;
            }
            self.spill((len * 2).max(4));
        }
        self.insert_spilled(value)
    }

    /// Moves the inline elements to a new heap allocation of `cap` elements.
    fn spill(&mut self, cap: usize) {
        debug_assert!(!self.is_spilled());
        let len = self.len();
        let heap = Self::allocate(cap, len);
        // SAFETY: the inline elements are initialized; the new allocation has room for `len`.
        unsafe {
            let dst = heap
                .as_ptr()
                .cast::<u8>()
                .add(Self::elements_offset())
                .cast::<T>();
            ptr::copy_nonoverlapping(self.as_ptr(), dst, len);
        }
        self.data = Data { heap };
        self.len |= SPILLED;
    }

    fn insert_spilled(&mut self, value: T) -> bool {
        let len = self.len();
        let main_len = self.main_len();
        let (main, tail) = self.runs();
        let main_pos = match main.binary_search(&value) {
            Ok(_) => return false,
            Err(pos) => pos,
        };
        let tail_pos = match tail.binary_search(&value) {
            Ok(_) => return false,
            Err(pos) => pos,
        };
        let appends_to_main = tail.is_empty() && main_pos == main_len;

        if len == self.heap_capacity() {
            // SAFETY: spilled, and the new capacity is larger than `len`.
            unsafe { self.set_capacity((len * 2).clamp(4, MAX_LEN.max(len + 1))) };
        }
        let ptr = self.as_mut_ptr();
        if appends_to_main {
            // SAFETY: `len < cap`; the tail is empty, so `len` is the end of the main run.
            unsafe {
                ptr.add(len).write(value);
                self.header_mut().main += 1;
            }
        } else {
            let at = main_len + tail_pos;
            // SAFETY: `len < cap`, so there is room to shift `at..len` up by one.
            unsafe {
                ptr::copy(ptr.add(at), ptr.add(at + 1), len - at);
                ptr.add(at).write(value);
            }
        }
        self.len += 1;

        let len = len + 1;
        if len - self.main_len() > MIN_TAIL.max(len.isqrt()) {
            self.merge_tail();
        }
        true
    }

    /// Merges the tail into the main run.
    fn merge_tail(&mut self) {
        if !self.is_spilled() || self.main_len() == self.len() {
            return;
        }
        let len = self.len();
        // SAFETY: the first `len` elements are initialized and owned by `self`.
        let elements = unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), len) };
        // Both runs are sorted and disjoint, which the stable sort merges in linear time.
        elements.sort();
        // SAFETY: spilled.
        unsafe { self.header_mut().main = len as u32 };
    }

    /// Removes `value`, returning whether it was in the set.
    pub fn remove(&mut self, value: &T) -> bool {
        let (main, tail) = self.runs();
        let main_len = main.len();
        let (at, in_main) = if let Ok(pos) = main.binary_search(value) {
            (pos, true)
        } else if let Ok(pos) = tail.binary_search(value) {
            (main_len + pos, false)
        } else {
            return false;
        };
        let len = self.len();
        let ptr = self.as_mut_ptr();
        // SAFETY: `at < len`; shifts the initialized `at + 1..len` down by one.
        unsafe { ptr::copy(ptr.add(at + 1), ptr.add(at), len - at - 1) };
        self.len -= 1;
        if in_main && self.is_spilled() {
            // SAFETY: spilled.
            unsafe { self.header_mut().main -= 1 };
        }
        true
    }

    pub fn extend(&mut self, iter: impl IntoIterator<Item = T>) {
        for value in iter {
            self.insert(value);
        }
    }

    /// Keeps only the elements for which `keep` returns true.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let len = self.len();
        let main_len = self.main_len();
        let ptr = self.as_mut_ptr();
        let (mut kept, mut kept_main) = (0, 0);
        for i in 0..len {
            // SAFETY: `i < len`, and `kept <= i`, so this moves initialized elements down.
            unsafe {
                let value = ptr.add(i).read();
                if keep(&value) {
                    ptr.add(kept).write(value);
                    kept += 1;
                    if i < main_len {
                        kept_main += 1;
                    }
                }
            }
        }
        self.len = kept as u32 | (self.len & SPILLED);
        if self.is_spilled() {
            // SAFETY: spilled.
            unsafe { self.header_mut().main = kept_main as u32 };
        }
    }

    /// Merges the tail and drops spare capacity, moving the elements inline if they fit.
    pub fn shrink_to_fit(&mut self) {
        if !self.is_spilled() {
            return;
        }
        self.merge_tail();
        let len = self.len();
        if len <= INLINE {
            let mut inline = [const { MaybeUninit::uninit() }; INLINE];
            // SAFETY: copies the `len` initialized elements out before freeing the allocation.
            unsafe {
                ptr::copy_nonoverlapping(self.as_ptr(), inline.as_mut_ptr().cast::<T>(), len);
                self.deallocate();
            }
            self.data = Data {
                inline: ManuallyDrop::new(inline),
            };
            self.len = len as u32;
        } else if self.heap_capacity() > len {
            // SAFETY: spilled, and `len <= cap`.
            unsafe { self.set_capacity(len) };
        }
    }
}

impl<T, const INLINE: usize> Drop for SortedSet<T, INLINE> {
    fn drop(&mut self) {
        // The elements are `Copy` (every way to add one requires it), so only the allocation needs
        // freeing.
        if self.is_spilled() {
            // SAFETY: spilled; `self` isn't used afterwards.
            unsafe { self.deallocate() };
        }
    }
}

impl<T: Copy + Ord, const INLINE: usize> Clone for SortedSet<T, INLINE> {
    fn clone(&self) -> Self {
        let mut elements = self.as_slice().to_vec();
        if self.main_len() != elements.len() {
            elements.sort();
        }
        Self::from_sorted_vec(elements)
    }
}

impl<T: Debug, const INLINE: usize> Debug for SortedSet<T, INLINE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.as_slice()).finish()
    }
}

impl<T: Copy + Ord, const INLINE: usize> PartialEq for SortedSet<T, INLINE> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|value| other.contains(value))
    }
}

impl<T: Copy + Ord, const INLINE: usize> Eq for SortedSet<T, INLINE> {}

impl<T: Copy + Ord, const INLINE: usize> FromIterator<T> for SortedSet<T, INLINE> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut elements: Vec<T> = iter.into_iter().collect();
        elements.sort_unstable();
        elements.dedup();
        Self::from_sorted_vec(elements)
    }
}

impl<T: Copy + Ord, const N: usize, const INLINE: usize> From<[T; N]> for SortedSet<T, INLINE> {
    fn from(values: [T; N]) -> Self {
        values.into_iter().collect()
    }
}

impl<'a, T, const INLINE: usize> IntoIterator for &'a SortedSet<T, INLINE> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Copy, const INLINE: usize> IntoIterator for SortedSet<T, INLINE> {
    type Item = T;
    type IntoIter = IntoIter<T, INLINE>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { set: self, next: 0 }
    }
}

/// An owning iterator over a [`SortedSet`]'s elements, in storage order.
pub struct IntoIter<T, const INLINE: usize> {
    set: SortedSet<T, INLINE>,
    next: usize,
}

impl<T: Copy, const INLINE: usize> Iterator for IntoIter<T, INLINE> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let value = self.set.as_slice().get(self.next).copied();
        self.next += 1;
        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.set.len().saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

impl<T: Copy, const INLINE: usize> ExactSizeIterator for IntoIter<T, INLINE> {}

impl<T: Copy + Ord, const INLINE: usize> ShrinkToFit for SortedSet<T, INLINE> {
    fn shrink_to_fit(&mut self) {
        SortedSet::shrink_to_fit(self);
    }
}

impl<T: Encode, const INLINE: usize> Encode for SortedSet<T, INLINE> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.len().encode(encoder)?;
        for value in self.as_slice() {
            value.encode(encoder)?;
        }
        Ok(())
    }
}

impl<Context, T: Decode<Context> + Copy + Ord, const INLINE: usize> Decode<Context>
    for SortedSet<T, INLINE>
{
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len = usize::decode(decoder)?;
        if len > MAX_LEN {
            return Err(DecodeError::Other("SortedSet too large"));
        }
        decoder.claim_container_read::<T>(len)?;
        let mut elements = Vec::with_capacity(len);
        for _ in 0..len {
            elements.push(T::decode(decoder)?);
        }
        // Sets are encoded in order, except for an unmerged tail or another set type's encoding.
        if !elements.is_sorted_by(|a, b| a < b) {
            elements.sort_unstable();
            elements.dedup();
        }
        Ok(Self::from_sorted_vec(elements))
    }
}

impl_borrow_decode!(SortedSet<T, INLINE>, T: Decode<__Context> + Copy + Ord, const INLINE: usize);

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// A small deterministic generator, so the tests run under Miri without extra dependencies.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self, bound: u32) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % bound as u64) as u32
        }
    }

    fn assert_matches<const INLINE: usize>(set: &SortedSet<u32, INLINE>, model: &BTreeSet<u32>) {
        assert_eq!(set.len(), model.len());
        let mut elements = set.as_slice().to_vec();
        elements.sort_unstable();
        assert!(elements.iter().eq(model.iter()));
        let (main, tail) = set.runs();
        assert!(main.is_sorted_by(|a, b| a < b));
        assert!(tail.is_sorted_by(|a, b| a < b));
    }

    fn random_ops<const INLINE: usize>(seed: u64, ops: usize, range: u32) {
        let mut rng = Lcg(seed);
        let mut set = SortedSet::<u32, INLINE>::new();
        let mut model = BTreeSet::new();
        for _ in 0..ops {
            let value = rng.next(range);
            match rng.next(10) {
                0..6 => assert_eq!(set.insert(value), model.insert(value)),
                6..8 => assert_eq!(set.remove(&value), model.remove(&value)),
                8 => {
                    set.retain(|v| v % 7 != value % 7);
                    model.retain(|v| v % 7 != value % 7);
                }
                _ => set.shrink_to_fit(),
            }
            assert_eq!(set.contains(&value), model.contains(&value));
            assert_matches(&set, &model);
        }
        let cloned = set.clone();
        assert_eq!(cloned, set);
        assert_matches(&cloned, &model);
        set.shrink_to_fit();
        assert_eq!(set.main_len(), set.len());
        if set.len() > INLINE {
            assert_eq!(set.heap_slots(), set.len());
        } else {
            assert!(!set.is_spilled());
        }
        assert_matches(&set, &model);
        let mut owned: Vec<_> = set.into_iter().collect();
        owned.sort_unstable();
        assert!(owned.iter().eq(model.iter()));
    }

    #[test]
    fn matches_a_btree_set() {
        let ops = if cfg!(miri) { 400 } else { 20_000 };
        for seed in 0..4 {
            random_ops::<0>(seed, ops, 300);
            random_ops::<3>(seed, ops, 40);
            random_ops::<6>(seed, ops, 2000);
        }
    }

    #[test]
    fn ascending_inserts_extend_the_main_run() {
        let mut set = SortedSet::<u32, 3>::new();
        for i in 0..1000 {
            assert!(set.insert(i));
        }
        assert_eq!(set.main_len(), 1000);
        assert!(set.as_slice().iter().copied().eq(0..1000));
    }

    #[test]
    fn tail_stays_short() {
        let mut set = SortedSet::<u32, 0>::new();
        let n = if cfg!(miri) { 2_000 } else { 100_000 };
        for i in (0..n).rev() {
            set.insert(i);
            assert!(set.len() - set.main_len() <= MIN_TAIL.max(set.len().isqrt()));
        }
        set.shrink_to_fit();
        assert!(set.as_slice().iter().copied().eq(0..n));
        assert_eq!(
            set.heap_size(),
            SortedSet::<u32, 0>::layout(n as usize).0.size()
        );
    }

    #[test]
    fn shrinks_back_inline() {
        let mut set: SortedSet<u32, 4> = (0..100).collect();
        set.retain(|&v| v < 3);
        set.shrink_to_fit();
        assert!(!set.is_spilled());
        assert_eq!(set.as_slice(), &[0, 1, 2]);
        assert_eq!(set.heap_size(), 0);
    }

    #[test]
    fn same_size_as_an_auto_set() {
        assert_eq!(size_of::<SortedSet<u32, 6>>(), 32);
        assert_eq!(size_of::<SortedSet<u64, 3>>(), 32);
        assert_eq!(size_of::<SortedSet<u32, 0>>(), 16);
    }

    #[test]
    fn encodes_like_an_auto_set() {
        let config = bincode::config::standard();
        let mut set = SortedSet::<u32, 2>::new();
        for v in [5, 3, 9, 1, 7] {
            set.insert(v);
        }
        set.shrink_to_fit();
        let encoded = bincode::encode_to_vec(&set, config).unwrap();
        let auto: crate::AutoSet<u32> = [5, 3, 9, 1, 7].into_iter().collect();
        let (decoded_auto, _): (crate::AutoSet<u32>, _) =
            bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(decoded_auto, auto);
        let from_auto = bincode::encode_to_vec(&auto, config).unwrap();
        let (decoded, _): (SortedSet<u32, 2>, _) =
            bincode::decode_from_slice(&from_auto, config).unwrap();
        assert_eq!(decoded, set);
        assert_eq!(decoded.as_slice(), &[1, 3, 5, 7, 9]);
    }
}

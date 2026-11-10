//! A set of `Copy + Ord` values stored as a sorted array once built, for sets that are built once
//! and then mostly read, like a task's forward dependency edges.
//!
//! A finished set stores exactly its elements: no empty buckets, control bytes or power-of-two
//! rounding as in a hash table. Lookups are binary searches, and iteration is in order.
//!
//! # Representation
//! * Up to `INLINE` elements are stored inline, sorted.
//! * Past that, the elements live in a heap allocation behind a thin pointer, after a small header
//!   holding the capacity and the length of the sorted *main* run. The elements are the main run
//!   followed by a short sorted *tail* of recent insertions, merged into the main run once full.
//!   Inserting in ascending order, the common case for freshly allocated ids, extends the main run
//!   directly.
//! * A sorted array is a poor structure to build a large set in random order: each insertion moves
//!   elements. So a set that is still growing past 32 elements (`MAX_MERGED_LEN`) switches to a
//!   hash table, and [`SortedSet::shrink_to_fit`] (called once the set is complete) sorts it back
//!   into an exactly sized array. Building a set then costs about as much as building a hash set,
//!   plus one sort.
//! * `shrink_to_fit` also merges the tail and drops spare capacity. A set shrunk to at most
//!   `INLINE` elements moves back inline.
//!
//! The encoded form (the length, then the elements) is the same as an [`AutoSet`]'s.
//!
//! [`AutoSet`]: crate::AutoSet

use std::{
    alloc::{self, Layout},
    fmt::{self, Debug},
    hash::{BuildHasherDefault, Hash},
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
use rustc_hash::FxHasher;
use shrink_to_fit::ShrinkToFit;

/// The hash table form. A map, as only hashbrown's maps report their allocation size.
type HashSet<T> = hashbrown::HashMap<T, (), BuildHasherDefault<FxHasher>>;

/// Set in [`SortedSet::len`] while the elements live in a sorted heap allocation.
const SPILLED: u32 = 1 << 31;
/// Set in [`SortedSet::len`] while the elements live in a hash table.
const HASHED: u32 = 1 << 30;
/// The largest number of elements a set can hold.
const MAX_LEN: usize = (HASHED - 1) as usize;
/// The tail is merged into the main run once longer than this.
const MAX_TAIL: usize = 16;
/// A sorted set whose tail is full switches to a hash table instead of merging once it holds more
/// than this many elements, as merging moves the whole array.
const MAX_MERGED_LEN: usize = 32;

pub struct SortedSet<T, const INLINE: usize = 0> {
    /// The number of inline or sorted elements, with [`SPILLED`] set while they live in a sorted
    /// heap allocation. Just [`HASHED`] while they live in a hash table.
    len: u32,
    data: Data<T, INLINE>,
}

union Data<T, const INLINE: usize> {
    inline: ManuallyDrop<[MaybeUninit<T>; INLINE]>,
    /// Valid while [`SPILLED`]: a [`Header`] followed by `cap` element slots.
    heap: NonNull<Header>,
    /// Valid while [`HASHED`]: a boxed hash table.
    table: NonNull<HashSet<T>>,
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

/// An iterator over a [`SortedSet`]'s elements: in order once the set is complete.
pub enum Iter<'a, T> {
    Slice(slice::Iter<'a, T>),
    Table(hashbrown::hash_map::Keys<'a, T, ()>),
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        match self {
            Iter::Slice(iter) => iter.next(),
            Iter::Table(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Iter::Slice(iter) => iter.size_hint(),
            Iter::Table(iter) => iter.size_hint(),
        }
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Iter::Slice(iter) => Iter::Slice(iter.clone()),
            Iter::Table(iter) => Iter::Table(iter.clone()),
        }
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
    const fn is_spilled(&self) -> bool {
        self.len & SPILLED != 0
    }

    #[inline]
    const fn is_hashed(&self) -> bool {
        self.len & HASHED != 0
    }

    /// Whether the elements are in a hash table, i.e. the set grew large and isn't complete yet.
    pub const fn is_hash_table(&self) -> bool {
        self.is_hashed()
    }

    fn table(&self) -> Option<&HashSet<T>> {
        if self.is_hashed() {
            // SAFETY: while hashed, `table` points to a live boxed table owned by `self`. Checked
            // first: the field is only valid while hashed.
            Some(unsafe { self.data.table.as_ref() })
        } else {
            None
        }
    }

    fn table_mut(&mut self) -> Option<&mut HashSet<T>> {
        if self.is_hashed() {
            // SAFETY: see `table`.
            Some(unsafe { self.data.table.as_mut() })
        } else {
            None
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self.table() {
            Some(table) => table.len(),
            None => (self.len & !SPILLED) as usize,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// The first inline or sorted element. Must not be hashed.
    fn as_ptr(&self) -> *const T {
        debug_assert!(!self.is_hashed());
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
            // SAFETY: not spilled or hashed, so `inline` is the active field.
            unsafe { self.data.inline.as_ptr().cast::<T>() }
        }
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        debug_assert!(!self.is_hashed());
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

    /// The inline or sorted elements: the sorted main run followed by the sorted tail. Must not be
    /// hashed.
    fn sorted_slice(&self) -> &[T] {
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

    /// The sorted main run and the sorted tail. Must not be hashed.
    fn runs(&self) -> (&[T], &[T]) {
        self.sorted_slice().split_at(self.main_len())
    }

    pub fn iter(&self) -> Iter<'_, T> {
        match self.table() {
            Some(table) => Iter::Table(table.keys()),
            None => Iter::Slice(self.sorted_slice().iter()),
        }
    }

    /// The capacity of the sorted heap allocation, or `0` otherwise.
    fn heap_capacity(&self) -> usize {
        if self.is_spilled() {
            // SAFETY: spilled.
            unsafe { self.header().cap as usize }
        } else {
            0
        }
    }

    /// Entry slots in the heap allocation: the sorted capacity or the hash table's buckets.
    pub fn heap_slots(&self) -> usize {
        match self.table() {
            Some(table) => match table.capacity() {
                0 => 0,
                capacity if capacity < 8 => capacity + 1,
                capacity => capacity / 7 * 8,
            },
            None => self.heap_capacity(),
        }
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

    /// Frees the heap allocation or hash table, without touching inline elements (they are `Copy`,
    /// see the impls). Afterwards `data` dangles until replaced.
    fn deallocate(&mut self) {
        if self.is_hashed() {
            // SAFETY: hashed, so `table` came from `Box::into_raw` and is owned by `self`.
            drop(unsafe { Box::from_raw(self.data.table.as_ptr()) });
        } else if self.is_spilled() {
            // SAFETY: spilled, so `heap` was allocated by `allocate`/`set_capacity` with this
            // layout.
            unsafe {
                let (layout, _) = Self::layout(self.heap_capacity());
                alloc::dealloc(self.data.heap.as_ptr().cast(), layout);
            }
        }
    }

    /// Resizes the sorted heap allocation to `cap` elements.
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
        self.deallocate();
        self.len = 0;
        self.data = Data {
            inline: ManuallyDrop::new([const { MaybeUninit::uninit() }; INLINE]),
        };
    }
}

impl<T: Copy + Ord + Hash, const INLINE: usize> SortedSet<T, INLINE> {
    /// Bytes this set owns on the heap. Excludes heap memory owned by the elements.
    pub fn heap_size(&self) -> usize {
        if let Some(table) = self.table() {
            size_of::<HashSet<T>>() + table.allocation_size()
        } else if self.is_spilled() {
            Self::layout(self.heap_capacity()).0.size()
        } else {
            0
        }
    }

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
        if let Some(table) = self.table() {
            return table.contains_key(value);
        }
        let (main, tail) = self.runs();
        main.binary_search(value).is_ok() || tail.binary_search(value).is_ok()
    }

    /// Inserts `value`, returning whether it was not in the set yet.
    pub fn insert(&mut self, value: T) -> bool {
        if let Some(table) = self.table_mut() {
            return table.insert(value, ()).is_none();
        }
        let len = self.len();
        if !self.is_spilled() {
            let pos = match self.sorted_slice().binary_search(&value) {
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
        debug_assert!(!self.is_spilled() && !self.is_hashed());
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
        // Ascending insertions extend the main run without a search.
        let appends_to_main = tail.is_empty() && main.last().is_none_or(|last| *last < value);
        let tail_pos = if appends_to_main {
            0
        } else {
            if main.binary_search(&value).is_ok() {
                return false;
            }
            match tail.binary_search(&value) {
                Ok(_) => return false,
                Err(pos) => pos,
            }
        };
        if !appends_to_main && len - main_len >= MAX_TAIL {
            if len >= MAX_MERGED_LEN {
                self.convert_to_table();
                return self.insert(value);
            }
            self.merge_tail();
            return self.insert_spilled(value);
        }

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
        true
    }

    /// Merges the tail into the main run. Must not be hashed.
    fn merge_tail(&mut self) {
        if !self.is_spilled() || self.main_len() == self.len() {
            return;
        }
        let len = self.len();
        // SAFETY: the first `len` elements are initialized and owned by `self`.
        let elements = unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), len) };
        // The runs are disjoint, so the unstable sort's result is the same as a stable one's.
        elements.sort_unstable();
        // SAFETY: spilled.
        unsafe { self.header_mut().main = len as u32 };
    }

    /// Moves the sorted elements into a hash table.
    fn convert_to_table(&mut self) {
        let mut table = HashSet::with_capacity_and_hasher(self.len() * 2, Default::default());
        table.extend(self.sorted_slice().iter().map(|&value| (value, ())));
        self.deallocate();
        self.data = Data {
            table: NonNull::from(Box::leak(Box::new(table))),
        };
        self.len = HASHED;
    }

    /// Removes `value`, returning whether it was in the set.
    pub fn remove(&mut self, value: &T) -> bool {
        if let Some(table) = self.table_mut() {
            return table.remove(value).is_some();
        }
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
        if let Some(table) = self.table_mut() {
            table.retain(|value, ()| keep(value));
            return;
        }
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

    /// Sorts the set into an exactly sized array (inline if it fits), for once it is complete.
    pub fn shrink_to_fit(&mut self) {
        if let Some(table) = self.table() {
            let mut elements: Vec<T> = table.keys().copied().collect();
            elements.sort_unstable();
            *self = Self::from_sorted_vec(elements);
            return;
        }
        if !self.is_spilled() {
            return;
        }
        self.merge_tail();
        let len = self.len();
        if len <= INLINE {
            let mut inline = [const { MaybeUninit::uninit() }; INLINE];
            // SAFETY: copies the `len` initialized elements out before freeing the allocation.
            unsafe {
                ptr::copy_nonoverlapping(self.as_ptr(), inline.as_mut_ptr().cast::<T>(), len)
            };
            self.deallocate();
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
        // Inline and sorted elements are `Copy` (every way to add one requires it), so only the
        // allocation needs freeing; a hash table drops its own elements.
        self.deallocate();
    }
}

impl<T: Copy + Ord + Hash, const INLINE: usize> Clone for SortedSet<T, INLINE> {
    fn clone(&self) -> Self {
        let mut elements: Vec<T> = self.iter().copied().collect();
        if self.is_hashed() || self.main_len() != elements.len() {
            elements.sort_unstable();
        }
        Self::from_sorted_vec(elements)
    }
}

impl<T: Debug, const INLINE: usize> Debug for SortedSet<T, INLINE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T: Copy + Ord + Hash, const INLINE: usize> PartialEq for SortedSet<T, INLINE> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|value| other.contains(value))
    }
}

impl<T: Copy + Ord + Hash, const INLINE: usize> Eq for SortedSet<T, INLINE> {}

impl<T: Copy + Ord + Hash, const INLINE: usize> FromIterator<T> for SortedSet<T, INLINE> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut elements: Vec<T> = iter.into_iter().collect();
        elements.sort_unstable();
        elements.dedup();
        Self::from_sorted_vec(elements)
    }
}

impl<T: Copy + Ord + Hash, const N: usize, const INLINE: usize> From<[T; N]>
    for SortedSet<T, INLINE>
{
    fn from(values: [T; N]) -> Self {
        values.into_iter().collect()
    }
}

impl<'a, T, const INLINE: usize> IntoIterator for &'a SortedSet<T, INLINE> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

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

/// An owning iterator over a [`SortedSet`]'s elements.
pub struct IntoIter<T, const INLINE: usize> {
    set: SortedSet<T, INLINE>,
    /// The index of the next element, or the number of elements yielded from a hash table.
    next: usize,
}

impl<T: Copy, const INLINE: usize> Iterator for IntoIter<T, INLINE> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let value = match self.set.table() {
            // `nth` restarts a fresh iterator; fine for the rare owning iteration of a set that
            // is still a hash table.
            Some(table) => table.keys().nth(self.next).copied(),
            None => self.set.sorted_slice().get(self.next).copied(),
        };
        self.next += 1;
        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.set.len().saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

impl<T: Copy, const INLINE: usize> ExactSizeIterator for IntoIter<T, INLINE> {}

impl<T: Copy + Ord + Hash, const INLINE: usize> ShrinkToFit for SortedSet<T, INLINE> {
    fn shrink_to_fit(&mut self) {
        SortedSet::shrink_to_fit(self);
    }
}

impl<T: Encode, const INLINE: usize> Encode for SortedSet<T, INLINE> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.len().encode(encoder)?;
        for value in self.iter() {
            value.encode(encoder)?;
        }
        Ok(())
    }
}

impl<Context, T: Decode<Context> + Copy + Ord + Hash, const INLINE: usize> Decode<Context>
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
        // Complete sets are encoded in order; others (an unmerged tail, a hash table, another set
        // type's encoding) are sorted here.
        if !elements.is_sorted_by(|a, b| a < b) {
            elements.sort_unstable();
            elements.dedup();
        }
        Ok(Self::from_sorted_vec(elements))
    }
}

impl_borrow_decode!(SortedSet<T, INLINE>, T: Decode<__Context> + Copy + Ord + Hash, const INLINE: usize);

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
        let mut elements: Vec<u32> = set.iter().copied().collect();
        elements.sort_unstable();
        assert!(elements.iter().eq(model.iter()));
        if !set.is_hashed() {
            let (main, tail) = set.runs();
            assert!(main.is_sorted_by(|a, b| a < b));
            assert!(tail.is_sorted_by(|a, b| a < b));
            assert!(tail.len() <= MAX_TAIL);
        }
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
                _ => {
                    if rng.next(4) == 0 {
                        set.shrink_to_fit();
                    }
                }
            }
            assert_eq!(set.contains(&value), model.contains(&value));
            assert_matches(&set, &model);
        }
        let cloned = set.clone();
        assert_eq!(cloned, set);
        assert_matches(&cloned, &model);
        let mut owned: Vec<_> = set.clone().into_iter().collect();
        owned.sort_unstable();
        assert!(owned.iter().eq(model.iter()));
        set.shrink_to_fit();
        assert!(!set.is_hashed());
        assert_eq!(set.main_len(), set.len());
        if set.len() > INLINE {
            assert_eq!(set.heap_slots(), set.len());
        } else {
            assert!(!set.is_spilled());
        }
        assert!(set.iter().copied().eq(model.iter().copied()));
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
        assert!(!set.is_hashed());
        assert_eq!(set.main_len(), 1000);
        assert!(set.iter().copied().eq(0..1000));
    }

    #[test]
    fn large_random_sets_are_hashed_until_shrunk() {
        let mut set = SortedSet::<u32, 0>::new();
        let n = if cfg!(miri) { 500 } else { 50_000 };
        for i in (0..n).rev() {
            set.insert(i);
        }
        assert!(set.is_hashed());
        assert_eq!(set.len(), n as usize);
        set.shrink_to_fit();
        assert!(!set.is_hashed());
        assert!(set.iter().copied().eq(0..n));
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
        assert_eq!(set.sorted_slice(), &[0, 1, 2]);
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
        assert_eq!(decoded.sorted_slice(), &[1, 3, 5, 7, 9]);
    }
}

//! The slice of LightGBM's C API that training needs, behind safe handles.
//!
//! `lightgbm3` wraps prediction well but training only partly: it cannot set
//! query groups, which lambdarank cannot do without, and it cannot attach a
//! validation set, which early stopping cannot do without. So training talks
//! to `lightgbm3-sys` - the same statically linked LightGBM that `lightgbm3`
//! predicts with - directly, and every `unsafe` block in training lives here.
//!
//! Python's `lgb.train` is a loop over these same calls (`LGBM_BoosterCreate`,
//! `LGBM_BoosterAddValidData`, `LGBM_BoosterUpdateOneIter`,
//! `LGBM_BoosterGetEval`); early stopping is a callback in the Python wrapper,
//! not in the library, which is why `super` reimplements it.

use anyhow::{Result, bail};
use lightgbm3_sys as sys;
use std::ffi::{CStr, CString, c_int, c_void};
use std::marker::PhantomData;

/// Turn a C API return code into a `Result`, with LightGBM's own message.
fn check(code: c_int) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: LightGBM returns a pointer to a thread-local, NUL-terminated
    // buffer holding the last error, valid until the next API call.
    let message = unsafe { CStr::from_ptr(sys::LGBM_GetLastError()) };
    bail!("LightGBM: {}", message.to_string_lossy())
}

fn to_c_int(n: usize, what: &str) -> Result<c_int> {
    c_int::try_from(n).map_err(|_| anyhow::anyhow!("{what} {n} does not fit a C int"))
}

/// Rows, their labels, weights and episode groups, binned by LightGBM.
///
/// LightGBM copies the matrix and the fields while building its histograms, so
/// none of the slices handed in have to outlive this.
pub struct Dataset {
    handle: sys::DatasetHandle,
}

/// What goes into a [`Dataset`]: a row-major matrix and one value per row.
pub struct DatasetInput<'a> {
    pub features: &'a [f64],
    pub num_features: usize,
    pub labels: &'a [f32],
    pub weights: &'a [f32],
    /// Rows per episode, in row order. lambdarank compares rows only within
    /// a group.
    pub group_sizes: &'a [i32],
    pub feature_names: &'a [&'a str],
}

impl Dataset {
    /// Build a dataset. `reference` is the training set, when this is a
    /// validation set: it must be binned with the training set's bin
    /// boundaries, or its scores mean nothing.
    pub fn new(input: &DatasetInput, params: &str, reference: Option<&Dataset>) -> Result<Self> {
        let rows = input.labels.len();
        assert!(input.num_features > 0, "a dataset needs features");
        assert_eq!(
            input.features.len(),
            rows * input.num_features,
            "the matrix must be rows x features"
        );
        assert_eq!(input.weights.len(), rows, "one weight per row");
        assert_eq!(
            input.group_sizes.iter().map(|&g| g as usize).sum::<usize>(),
            rows,
            "the groups must cover every row exactly once"
        );
        assert_eq!(
            input.feature_names.len(),
            input.num_features,
            "one name per feature"
        );

        let params = CString::new(params)?;
        let mut handle: sys::DatasetHandle = std::ptr::null_mut();
        // SAFETY: the matrix is `rows * num_features` f64s, row-major, as
        // asserted above; LightGBM copies it and keeps no pointer into it.
        check(unsafe {
            sys::LGBM_DatasetCreateFromMat(
                input.features.as_ptr().cast::<c_void>(),
                sys::C_API_DTYPE_FLOAT64 as c_int,
                to_c_int(rows, "row count")?,
                to_c_int(input.num_features, "feature count")?,
                1, // row-major
                params.as_ptr(),
                reference.map_or(std::ptr::null_mut(), |r| r.handle),
                &mut handle,
            )
        })?;
        // Owned from here, so an error below frees it.
        let dataset = Dataset { handle };

        dataset.set_field(
            c"label",
            input.labels.as_ptr().cast(),
            rows,
            sys::C_API_DTYPE_FLOAT32,
        )?;
        dataset.set_field(
            c"weight",
            input.weights.as_ptr().cast(),
            rows,
            sys::C_API_DTYPE_FLOAT32,
        )?;
        dataset.set_field(
            c"group",
            input.group_sizes.as_ptr().cast(),
            input.group_sizes.len(),
            sys::C_API_DTYPE_INT32,
        )?;

        let names: Vec<CString> = input
            .feature_names
            .iter()
            .map(|n| CString::new(*n))
            .collect::<Result<_, _>>()?;
        let mut name_ptrs: Vec<*const std::ffi::c_char> =
            names.iter().map(|n| n.as_ptr()).collect();
        // SAFETY: `name_ptrs` holds `num_features` pointers into `names`, which
        // outlives the call; LightGBM copies the strings.
        check(unsafe {
            sys::LGBM_DatasetSetFeatureNames(
                dataset.handle,
                name_ptrs.as_mut_ptr(),
                to_c_int(name_ptrs.len(), "feature name count")?,
            )
        })?;

        Ok(dataset)
    }

    fn set_field(&self, name: &CStr, data: *const c_void, len: usize, dtype: u32) -> Result<()> {
        // SAFETY: callers pass a pointer to `len` elements of `dtype`, which
        // LightGBM copies.
        check(unsafe {
            sys::LGBM_DatasetSetField(
                self.handle,
                name.as_ptr(),
                data,
                to_c_int(len, "field length")?,
                dtype as c_int,
            )
        })
    }
}

impl Drop for Dataset {
    fn drop(&mut self) {
        // SAFETY: the handle came from LGBM_DatasetCreateFromMat and is freed
        // once. Nothing useful can be done with an error while dropping.
        unsafe { sys::LGBM_DatasetFree(self.handle) };
    }
}

/// A model being trained. It borrows its datasets: LightGBM keeps pointers to
/// them for as long as the booster lives.
pub struct Booster<'data> {
    handle: sys::BoosterHandle,
    num_features: usize,
    _datasets: PhantomData<&'data Dataset>,
}

impl<'data> Booster<'data> {
    pub fn new(train: &'data Dataset, params: &str, num_features: usize) -> Result<Self> {
        let params = CString::new(params)?;
        let mut handle: sys::BoosterHandle = std::ptr::null_mut();
        // SAFETY: `train` is a live dataset, and the lifetime keeps it alive
        // for as long as the booster.
        check(unsafe { sys::LGBM_BoosterCreate(train.handle, params.as_ptr(), &mut handle) })?;
        Ok(Booster {
            handle,
            num_features,
            _datasets: PhantomData,
        })
    }

    /// Attach a validation set. It becomes data index 1 for [`Self::eval`];
    /// index 0 is the training set.
    pub fn add_valid(&mut self, valid: &'data Dataset) -> Result<()> {
        // SAFETY: both handles are live, and the lifetime keeps `valid` alive.
        check(unsafe { sys::LGBM_BoosterAddValidData(self.handle, valid.handle) })
    }

    /// Grow one more tree.
    ///
    /// LightGBM reports when it can find no split worth making; Python's
    /// `lgb.train` ignores that and keeps iterating, so this does too.
    pub fn update_one_iter(&mut self) -> Result<()> {
        let mut is_finished: c_int = 0;
        // SAFETY: live handle, and `is_finished` is a valid out pointer.
        check(unsafe { sys::LGBM_BoosterUpdateOneIter(self.handle, &mut is_finished) })
    }

    /// The metrics on dataset `data_idx`, in the order the `metric` and
    /// `eval_at` parameters list them.
    pub fn eval(&self, data_idx: usize) -> Result<Vec<f64>> {
        let mut count: c_int = 0;
        // SAFETY: live handle, valid out pointer.
        check(unsafe { sys::LGBM_BoosterGetEvalCounts(self.handle, &mut count) })?;
        let mut results = vec![0.0f64; count as usize];
        let mut written: c_int = 0;
        // SAFETY: `results` has room for every metric LightGBM said it has.
        check(unsafe {
            sys::LGBM_BoosterGetEval(
                self.handle,
                to_c_int(data_idx, "data index")?,
                &mut written,
                results.as_mut_ptr(),
            )
        })?;
        assert_eq!(
            written, count,
            "LightGBM wrote as many metrics as it counted"
        );
        Ok(results)
    }

    /// The model as text, every tree, with split-count importances - what
    /// Python's `Booster.save_model()` writes.
    pub fn save_model_to_string(&self) -> Result<String> {
        let mut len: i64 = 0;
        // First call: ask how long the string is.
        // SAFETY: a zero-length buffer is allowed; LightGBM only reports the
        // length it needs.
        check(unsafe {
            sys::LGBM_BoosterSaveModelToString(
                self.handle,
                0,
                -1,
                sys::C_API_FEATURE_IMPORTANCE_SPLIT as c_int,
                0,
                &mut len,
                std::ptr::null_mut(),
            )
        })?;
        let mut buffer = vec![0u8; len as usize];
        let mut written: i64 = 0;
        // SAFETY: `buffer` is exactly the length LightGBM asked for.
        check(unsafe {
            sys::LGBM_BoosterSaveModelToString(
                self.handle,
                0,
                -1,
                sys::C_API_FEATURE_IMPORTANCE_SPLIT as c_int,
                len,
                &mut written,
                buffer.as_mut_ptr().cast(),
            )
        })?;
        assert_eq!(
            written, len,
            "the model did not change between the two calls"
        );
        let text = CStr::from_bytes_until_nul(&buffer)?;
        Ok(text.to_str()?.to_owned())
    }

    /// Total gain of each feature's splits, over every tree, in feature order.
    pub fn gain_importance(&self) -> Result<Vec<f64>> {
        let mut importance = vec![0.0f64; self.num_features];
        // SAFETY: one f64 per feature, which is what LightGBM writes.
        check(unsafe {
            sys::LGBM_BoosterFeatureImportance(
                self.handle,
                0, // every iteration
                sys::C_API_FEATURE_IMPORTANCE_GAIN as c_int,
                importance.as_mut_ptr(),
            )
        })?;
        Ok(importance)
    }
}

impl Drop for Booster<'_> {
    fn drop(&mut self) {
        // SAFETY: the handle came from LGBM_BoosterCreate and is freed once.
        unsafe { sys::LGBM_BoosterFree(self.handle) };
    }
}

import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
import scipy.ndimage
import tempfile
import os
import gc
import base64
from PIL import Image
from scipy.ndimage import center_of_mass

# =============================
# Configuration
# =============================
PATCH_SIZE = (128, 128, 64)
MODEL_PATH = "/Users/alyaemara/Desktop/brain_tumor_app/model/unet3d_fold_0.keras"


# =============================
# Utility Functions
# =============================
def normalize_volume(vol):
    vol = vol.astype(np.float32)
    out = np.zeros_like(vol)
    for c in range(vol.shape[-1]):
        channel = vol[..., c]
        mask = channel > 0
        if np.any(mask):
            mean = channel[mask].mean()
            std = channel[mask].std()
            out[..., c] = (channel - mean) / (std + 1e-8)
        else:
            out[..., c] = channel
    return out


def get_brain_bbox(vol):
    mask = np.any(vol > 0, axis=-1)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, vol.shape[0], 0, vol.shape[1], 0, vol.shape[2]
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1
    return x0, x1, y0, y1, z0, z1


def crop_to_bbox(vol, bbox):
    x0, x1, y0, y1, z0, z1 = bbox
    return vol[x0:x1, y0:y1, z0:z1]


def overlay_mask_colored(image, mask, alpha=0.4):
    """Overlay multi-label mask with specific colors for NCR, ED, ET."""
    img = (image * 255).astype(np.uint8)
    img = np.stack([img] * 3, axis=-1)
    overlay = img.copy()

    # Label 1: Necrotic Core (Blue)
    overlay[mask == 1] = [0, 0, 255]
    # Label 2: Edema (Yellow)
    overlay[mask == 2] = [255, 255, 0]
    # Label 4: Enhancing Tumor (Red)
    overlay[mask == 4] = [255, 0, 0]

    return Image.fromarray((img * (1 - alpha) + overlay * alpha).astype(np.uint8))


# =============================
# Model Loader
# =============================
@st.cache_resource
def load_segmentation_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# =============================
# Streamlit UI Setup
# =============================
st.set_page_config(page_title="Brain Tumor AI", layout="wide")

# Logo Handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "photos", "PHOTO-2026-02-03-15-14-56.jpg")


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


try:
    logo_base64 = get_base64_of_bin_file(LOGO_PATH)
    logo_html = f'<img src="data:image/jpeg;base64,{logo_base64}" style="height: 110px; border-radius: 12px;">'
except:
    logo_html = ""

st.markdown(f"""
    <div style="background: linear-gradient(to right, #FF7F50, #FF8C00); padding: 1.5rem; border-radius: 15px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 25px;">
        <div>
            <h1 style='margin: 0; color:white; font-size: 2.8rem;'>🧠 Brain Tumor Segmentation AI</h1>
            <p style='margin: 0; color:white; opacity: 0.9; font-size: 1.2rem;'>Orange Digital Center - Instant AI Hackathon</p>
        </div>
        <div>{logo_html}</div>
    </div>
""", unsafe_allow_html=True)

# =============================
# Upload Section
# =============================
with st.expander("🧩 Step 1: Upload MRI Modalities (FLAIR, T1, T1ce, T2)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        flair_f = st.file_uploader("FLAIR", type=["nii", "nii.gz"])
        t1_f = st.file_uploader("T1", type=["nii", "nii.gz"])
    with col2:
        t1ce_f = st.file_uploader("T1ce", type=["nii", "nii.gz"])
        t2_f = st.file_uploader("T2", type=["nii", "nii.gz"])

# =============================
# Main Processing Logic
# =============================
if all([flair_f, t1_f, t1ce_f, t2_f]):
    if st.button("🚀 Run AI Segmentation Analysis", use_container_width=True):
        with st.spinner("Analyzing MRI volumes... this may take a moment."):
            def save_tmp(uploaded):
                suffix = ".nii.gz" if uploaded.name.endswith(".nii.gz") else ".nii"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(uploaded.getbuffer())
                return f.name


            paths = [save_tmp(f) for f in [flair_f, t1_f, t1ce_f, t2_f]]

            try:
                # Load and stack
                vols = [nib.load(p).get_fdata() for p in paths]
                X = np.stack(vols, axis=-1)
                orig_shape = X.shape[:-1]
                X_norm = normalize_volume(X)
                bbox = get_brain_bbox(X_norm)
                X_crop = crop_to_bbox(X_norm, bbox)

                # Resize for model
                zoom = np.array(PATCH_SIZE) / np.array(X_crop.shape[:-1])
                X_resized = scipy.ndimage.zoom(X_crop, np.append(zoom, 1.0), order=1)

                # Predict
                model = load_segmentation_model()
                preds = model.predict(X_resized[None, ...], verbose=0)
                seg = np.argmax(preds[0], axis=-1).astype(np.uint8)

                # Undo Resize
                inv_zoom = np.array(X_crop.shape[:-1]) / np.array(PATCH_SIZE)
                seg = scipy.ndimage.zoom(seg, inv_zoom, order=0)

                # Map back to full volume
                full_mask = np.zeros(orig_shape, dtype=np.uint8)
                x0, x1, y0, y1, z0, z1 = bbox
                d, h, w = full_mask[x0:x1, y0:y1, z0:z1].shape
                full_mask[x0:x1, y0:y1, z0:z1] = seg[:d, :h, :w]

                # Count Voxels
                ncr_voxels = int(np.sum(full_mask == 1))
                ed_voxels = int(np.sum(full_mask == 2))
                et_voxels = int(np.sum(full_mask == 4))
                total_tumor = ncr_voxels + ed_voxels + et_voxels

                # Results Header
                st.markdown("---")
                if total_tumor > 0:
                    st.error(f"### ⚠️ Analysis Complete: Tumor Detected ({total_tumor} total voxels)")

                    # Store results in session state for advice viewing
                    st.session_state['ncr'] = ncr_voxels
                    st.session_state['ed'] = ed_voxels
                    st.session_state['et'] = et_voxels
                    st.session_state['processed'] = True

                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Necrotic Core (NCR)", f"{ncr_voxels} vox", "Label 1")
                    m2.metric("Edema (ED)", f"{ed_voxels} vox", "Label 2")
                    m3.metric("Enhancing Tumor (ET)", f"{et_voxels} vox", "Label 4")
                else:
                    st.success("### ✅ Analysis Complete: No Tumor Detected")
                    st.session_state['processed'] = False

                # Large Visualization Section
                st.markdown("### 🔍 Interactive Scan Viewer")
                tumor_slices = np.where(np.sum(full_mask, axis=(0, 1)) > 0)[0]
                default_slice = int(tumor_slices[len(tumor_slices) // 2]) if total_tumor > 0 else 75

                slice_idx = st.slider("Navigate through Axial Slices", 0, orig_shape[2] - 1, default_slice)

                # Displaying Images Much Larger
                v_col1, v_col2 = st.columns([1, 1])


                # Normalization for display
                def norm_d(img):
                    return (img - img.min()) / (img.max() - img.min() + 1e-8)


                with v_col1:
                    st.image(norm_d(X[:, :, slice_idx, 0]), caption="FLAIR Modality (Raw)", use_container_width=True)
                with v_col2:
                    if total_tumor > 0:
                        ov = overlay_mask_colored(norm_d(X[:, :, slice_idx, 0]), full_mask[:, :, slice_idx])
                        st.image(ov, caption="AI Segmentation (Multi-Label)", use_container_width=True)
                    else:
                        st.info("No tumor detected to overlay.")

                # Legend for Colors
                if total_tumor > 0:
                    st.markdown("""
                        <div style='display: flex; justify-content: center; gap: 20px; background: #f0f2f6; padding: 10px; border-radius: 10px;'>
                            <span><b style='color:blue;'>■</b> Necrotic Core</span>
                            <span><b style='color:yellow;'>■</b> Edema (Swelling)</span>
                            <span><b style='color:red;'>■</b> Enhancing Tumor (Active)</span>
                        </div>
                    """, unsafe_allow_html=True)

                # New Advice Section (Immediately Visible)
                if total_tumor > 0:
                    st.markdown("---")
                    st.subheader("📋 Clinical Findings & Advice")
                    with st.expander("🚀 CLICK HERE FOR DETAILED MEDICAL ADVICE & NEXT STEPS", expanded=True):
                        st.write("### Patient Report Summary")
                        if ncr_voxels > 0:
                            st.write(
                                "- **NCR detected:** Indicates areas of non-perfusing, dead tissue within the tumor core.")
                        if ed_voxels > 0:
                            st.write(
                                "- **ED detected:** Peritumoral edema indicates significant swelling, often causing pressure-related symptoms.")
                        if et_voxels > 0:
                            st.write(
                                "- **ET detected:** Enhancing tumor tissue represents the most metabolically active and growing portion.")

                        st.warning("""
                        **Recommended Handling:**
                        1. **Consultation:** Please take these results to a Neuro-Oncologist or Neuro-Surgeon immediately.
                        2. **Pressure Management:** If Edema (ED) is high, monitor for headaches or neurological deficits.
                        3. **Biopsy/Surgery:** The Enhancing Tumor (ET) area is typically the primary target for surgical resection or biopsy.
                        """)
                        st.info(
                            "Note: This AI analysis is a screening tool. Final diagnosis must be made by a certified radiologist.")

            except Exception as e:
                st.error(f"❌ Critical Error: {e}")
            finally:
                for p in paths:
                    if os.path.exists(p): os.remove(p)
                gc.collect()

elif not all([flair_f, t1_f, t1ce_f, t2_f]):
    st.info("Please upload all 4 MRI modalities to begin the AI analysis.")
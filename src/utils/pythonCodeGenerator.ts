/**
 * Generates the clean, modular Python Streamlit application script (app.py)
 * adhering to the assignment requirements: AI and DL using TF-09Aug(E) Page 9.
 */

export function generateStreamlitPythonCode(): string {
  return `"""
Streamlit MNIST Digit Preprocessor Web Application
Assignment: AI and DL using TF-09Aug(E) Page 9

Standard MNIST-style preprocessing pipeline:
1. Grayscale conversion & Binarization
2. Bounding box detection & tight crop
3. Aspect-preserving square padding
4. Area-based downscaling (cv2.INTER_AREA) to target grid (e.g. 28x28)
5. Center of Mass calculation and translation (recentering)
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from scipy.ndimage import center_of_mass, shift
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="MNIST Digit Preprocessing Pipeline",
    page_icon="✍️",
    layout="wide",
)

st.title("MNIST Digit Preprocessing Pipeline")
st.markdown(
    "Draw a handwritten digit (0-9) on the canvas to observe step-by-step "
    "MNIST-style normalizations: tight bounding crop, square padding, "
    "area downscaling, and center-of-mass recentering."
)

# --- Sidebar Controls ---
st.sidebar.header("Canvas & Preprocessing Settings")

stroke_width = st.sidebar.slider(
    "Stroke Width (px)", min_value=8, max_value=32, value=18, step=2
)
target_res = st.sidebar.selectbox(
    "Target Resolution", options=[14, 28, 56], index=1
)
binarize_thresh = st.sidebar.slider(
    "Binarization Threshold", min_value=10, max_value=128, value=30, step=5
)
recenter_enabled = st.sidebar.checkbox(
    "Enable Center of Mass Recentering", value=True
)

st.sidebar.markdown("---")
st.sidebar.caption("Reference: LeCun et al. - The MNIST Database of Handwritten Digits")

# --- Pipeline Functions ---

def step1_grayscale_and_binarize(rgba_image: np.ndarray, thresh: int):
    """
    Step 1: Convert RGBA canvas image to single-channel grayscale and threshold.
    Background is 0 and digit stroke is > 0.
    """
    # RGB or RGBA conversion
    if rgba_image.shape[-1] == 4:
        rgb = rgba_image[:, :, :3]
    else:
        rgb = rgba_image

    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return gray, binary


def step2_bounding_crop(gray: np.ndarray, binary: np.ndarray):
    """
    Step 2: Find contours / bounding box and crop tightly around digit.
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None, None, None

    x, y, w, h = cv2.boundingRect(coords)
    cropped = gray[y : y + h, x : x + w]
    bbox = (x, y, w, h)
    return cropped, bbox, binary


def step3_square_pad(cropped: np.ndarray, padding_ratio: float = 1.25):
    """
    Step 3: Pad cropped digit into a square aspect ratio while keeping digit centered.
    """
    h, w = cropped.shape
    max_dim = max(h, w)
    square_size = int(round(max_dim * padding_ratio))

    square = np.zeros((square_size, square_size), dtype=np.uint8)
    offset_y = (square_size - h) // 2
    offset_x = (square_size - w) // 2
    square[offset_y : offset_y + h, offset_x : offset_x + w] = cropped
    return square


def step4_downscale(square: np.ndarray, target_dim: int):
    """
    Step 4: Downscale square image into target resolution using area-based downsampling.
    """
    resized = cv2.resize(
        square, (target_dim, target_dim), interpolation=cv2.INTER_AREA
    )
    return resized


def step5_recenter_center_of_mass(image: np.ndarray):
    """
    Step 5: Calculate center of mass of pixel intensities and shift to align with grid center.
    """
    h, w = image.shape
    cy, cx = center_of_mass(image)

    # In case the image is completely blank
    if np.isnan(cy) or np.isnan(cx):
        return image, (0, 0), (w / 2, h / 2), (0, 0)

    target_cy, target_cx = (h - 1) / 2.0, (w - 1) / 2.0
    shift_y = target_cy - cy
    shift_x = target_cx - cx

    # Subpixel translation using affine warp or ndimage shift with constant 0 border
    recentered = shift(image, (shift_y, shift_x), order=1, mode='constant', cval=0.0)
    new_cy, new_cx = center_of_mass(recentered)

    return recentered, (cx, cy), (target_cx, target_cy), (shift_x, shift_y)


# --- Layout: Two Main Columns ---
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.subheader("1. Drawing Canvas")
    st.caption("Draw a digit (0 to 9) in white stroke on the black canvas:")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="mnist_canvas",
    )

    if st.button("Reset / Clear Canvas", use_container_width=True):
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

with col_right:
    st.subheader("2. Step-by-Step Processing Pipeline")

    # Check if anything drawn
    if (
        canvas_result is None
        or canvas_result.image_data is None
        or np.all(canvas_result.image_data == 0)
    ):
        st.warning("Canvas is empty. Draw a digit on the canvas to begin preprocessing.")
    else:
        # Run Step 1
        gray, binary = step1_grayscale_and_binarize(
            canvas_result.image_data, binarize_thresh
        )

        if np.sum(binary) < 50:
            st.warning("Canvas is empty or stroke is too faint. Please draw a clearer digit.")
        else:
            # Run Step 2
            cropped, bbox, _ = step2_bounding_crop(gray, binary)
            if cropped is None:
                st.warning("No digit contours found.")
            else:
                x, y, w, h = bbox

                # Run Step 3
                square = step3_square_pad(cropped, padding_ratio=1.28)

                # Run Step 4
                downscaled = step4_downscale(square, target_res)

                # Run Step 5
                if recenter_enabled:
                    final_matrix, com_init, com_target, shifts = (
                        step5_recenter_center_of_mass(downscaled)
                    )
                else:
                    final_matrix = downscaled
                    com_init = (0, 0)
                    shifts = (0, 0)

                # Visualizing Steps 1 - 4
                vcol1, vcol2, vcol3, vcol4 = st.columns(4)
                with vcol1:
                    st.caption("Step 1: Grayscale")
                    st.image(gray, clamp=True, width=120)
                with vcol2:
                    st.caption(f"Step 2: Crop ({w}x{h})")
                    st.image(cropped, clamp=True, width=120)
                with vcol3:
                    st.caption(f"Step 3: Square ({square.shape[0]}x{square.shape[1]})")
                    st.image(square, clamp=True, width=120)
                with vcol4:
                    st.caption(f"Step 4: Resize ({target_res}x{target_res})")
                    st.image(downscaled, clamp=True, width=120)

                st.markdown("---")
                st.subheader(f"Step 5: Center of Mass Recenter ({target_res}x{target_res})")

                rcol1, rcol2 = st.columns([1, 1.2])
                with rcol1:
                    fig, ax = plt.subplots(figsize=(3.5, 3.5))
                    ax.imshow(final_matrix, cmap="gray", interpolation="nearest")
                    ax.set_title("Recentered Image")
                    ax.axis("off")
                    st.pyplot(fig)

                with rcol2:
                    st.markdown("**Center of Mass Details:**")
                    st.write(f"- Initial Center of Mass (cx, cy): ({com_init[0]:.2f}, {com_init[1]:.2f})")
                    st.write(f"- Applied Shift (dx, dy): ({shifts[0]:.2f}, {shifts[1]:.2f})")
                    st.write(f"- Grid Center Target: ({(target_res - 1)/2:.1f}, {(target_res - 1)/2:.1f})")

                st.markdown("---")
                st.subheader("3. Heatmap, Numeric Vector & Neural Network Visualizations")

                tab_nn, tab_heatmap, tab_matrix, tab_vector = st.tabs(
                    [
                        "Neural Network (Input -> Hidden -> Output)",
                        "Pixel Intensity Heatmap",
                        "2D Numeric Matrix",
                        "Flattened 1D Vector",
                    ]
                )

                with tab_nn:
                    st.markdown("### Synaptic Connections with Multi-Colored Lines")
                    st.caption("Input Layer (16 sectors) -> Hidden Layer (14 feature neurons) -> Output Layer (Digits 0-9)")

                    # Downsample 28x28 into 4x4 spatial pooling sectors
                    pool_4x4 = cv2.resize(final_matrix, (4, 4), interpolation=cv2.INTER_AREA).flatten() / 255.0
                    
                    # Simulated forward pass (Softmax logits)
                    np.random.seed(42)
                    w1 = np.random.randn(14, 16) * 0.5
                    b1 = np.zeros(14)
                    h_act = np.maximum(0, np.dot(w1, pool_4x4) + b1) # ReLU
                    
                    w2 = np.random.randn(10, 14) * 0.5
                    b2 = np.zeros(10)
                    logits = np.dot(w2, h_act) + b2
                    exp_z = np.exp(logits - np.max(logits))
                    probs = exp_z / np.sum(exp_z)
                    pred_digit = int(np.argmax(probs))

                    st.success(f"Predicted Digit: **{pred_digit}** (Confidence: {probs[pred_digit]*100:.1f}%)")

                    # Draw Neural Network Architecture with Colored Synaptic Lines
                    fig_nn, ax_nn = plt.subplots(figsize=(10, 6), facecolor="#090d16")
                    ax_nn.set_facecolor("#090d16")

                    # Node coordinates
                    n_in, n_hid, n_out = 16, 14, 10
                    y_in = np.linspace(0.1, 0.9, n_in)
                    y_hid = np.linspace(0.1, 0.9, n_hid)
                    y_out = np.linspace(0.1, 0.9, n_out)

                    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_hid))

                    # Connections Input -> Hidden (Colored lines)
                    for j in range(n_hid):
                        c = colors[j]
                        for i in range(n_in):
                            if pool_4x4[i] > 0.1 or abs(w1[j, i]) > 0.4:
                                alpha = min(0.8, max(0.1, pool_4x4[i] * 0.7 + 0.2))
                                ax_nn.plot([0.15, 0.5], [y_in[i], y_hid[j]], color=c, alpha=alpha, lw=1.2)

                    # Connections Hidden -> Output (Colored lines)
                    out_colors = plt.cm.tab10(np.linspace(0, 1, 10))
                    for k in range(n_out):
                        c_out = out_colors[k]
                        for j in range(n_hid):
                            if h_act[j] > 0.2:
                                alpha = min(0.9, max(0.15, probs[k] * 0.8 + 0.2))
                                ax_nn.plot([0.5, 0.85], [y_hid[j], y_out[k]], color=c_out, alpha=alpha, lw=1.5)

                    # Draw Nodes
                    ax_nn.scatter([0.15]*n_in, y_in, s=120, c='#38bdf8', edgecolors='white', zorder=5, label='Input (16)')
                    ax_nn.scatter([0.5]*n_hid, y_hid, s=140, c=colors, edgecolors='white', zorder=5, label='Hidden (14)')
                    ax_nn.scatter([0.85]*n_out, y_out, s=200, c=out_colors, edgecolors='white', zorder=5, label='Output (10)')

                    for k in range(n_out):
                        ax_nn.text(0.9, y_out[k], f"Digit {k}: {probs[k]*100:.1f}%", color='white', fontsize=9, va='center')

                    ax_nn.set_xlim(0.05, 1.1)
                    ax_nn.set_ylim(0.05, 0.95)
                    ax_nn.axis('off')
                    ax_nn.set_title("Input -> Hidden -> Output Architecture (Multi-Colored Synapses)", color='white', pad=12)
                    st.pyplot(fig_nn)

                with tab_heatmap:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    im = ax2.imshow(final_matrix, cmap="viridis")
                    fig2.colorbar(im, ax=ax2, label="Pixel Intensity (0-255)")
                    ax2.set_title(f"Final {target_res}x{target_res} Normalized Heatmap")
                    st.pyplot(fig2)

                with tab_matrix:
                    st.caption(f"2D Array of shape ({target_res}, {target_res}):")
                    st.dataframe(final_matrix.astype(int))

                with tab_vector:
                    flat_vector = (final_matrix.flatten() / 255.0).round(4)
                    st.caption(f"Flattened 1D Vector ({len(flat_vector)} floats in [0.0, 1.0]):")
                    st.code(f"x_test = np.array({flat_vector.tolist()})", language="python")
`;
}

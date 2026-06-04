import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_highly_defined_petal(center_angle, length, width_scale, tip_curl, ripple_freq, tilt_angle, is_inner_tier, micro_lift, radial_offset):
    """
    Sculpts a highly chiseled teardrop data petal.
    """
    u = np.linspace(0, 1, 35)
    v = np.linspace(0, 1, 35) 
    U, V = np.meshgrid(u, v)
    
    # --- Concentric Layer Base Adjustments ---
    if is_inner_tier:
        length *= 1.00        # Changed from 0.82
        width_scale *= 1.00   # Changed from 0.75
        lift = 0.35          
    else:
        length *= 1.00        # Changed from 1.15
        width_scale *= 1.00   # Changed from 1.00
        lift = 0.0          
    
    r = (length * U) + radial_offset 
    
    # --- CHISELED TEARDROP PROFILE ---
    base_taper = U
    shoulder = np.sin(U * np.pi) ** 0.5
    tip_taper = (1.25 - U)
    
    edge_shape = np.clip(1 - V**3, 0, 1)
    base_width = width_scale * 1.4 * base_taper * shoulder * tip_taper * edge_shape
    
    # Smooth, tip-damped edge ripples
    edge_ripple = 0.05 * np.sin(ripple_freq * np.pi * U) * edge_shape * (U**2) * (1 - U)
    W = base_width + edge_ripple
    
    # --- EXPLICIT SYMMETRICAL MIRRORING (Local Space) ---
    x_left = r
    y_left = -W
    x_right = r
    y_right = W
    
    X_local = np.hstack([np.fliplr(x_left), x_right])
    Y_local = np.hstack([np.fliplr(y_left), y_right])
    
    # --- ENHANCED STRUCTURAL SHADING (Z-Axis) ---
    z_spine = 0.75 * np.sin(U * np.pi * 0.55)
    petal_cup = 0.28 * (1 - V**2) * np.sin(U * np.pi)
    center_ridge = -0.08 * np.abs(V) * np.sin(U * np.pi)
    tip_fold = tip_curl * (U**3.0)
    
    Z_half = z_spine + petal_cup + center_ridge - tip_fold
    Z_local = np.hstack([np.fliplr(Z_half), Z_half])
    
    # STEP 1 - LOCAL PITCH TILT
    x_pitched = X_local
    y_pitched = Y_local * np.cos(tilt_angle) - Z_local * np.sin(tilt_angle)
    z_pitched = Y_local * np.sin(tilt_angle) + Z_local * np.cos(tilt_angle)

    # STEP 2 - GLOBAL ROTATION MATRIX
    X_final = x_pitched * np.cos(center_angle) - y_pitched * np.sin(center_angle)
    Y_final = x_pitched * np.sin(center_angle) + y_pitched * np.cos(center_angle)
    Z_final = z_pitched
    
    # Apply baseline tier elevation
    Z_final += lift
    if micro_lift:
        Z_final += 0.12
        
    return X_final, Y_final, Z_final

def add_stamen_cluster(fig, count=18, max_height=1.15):
    """
    Injects an upright, gold-anthered stamen array into the core floor.
    """
    t = np.linspace(0, 1, 25)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    
    phi = np.linspace(0, 2 * np.pi, 16) 
    radius_bead = 0.024 
    
    for idx, angle in enumerate(angles):
        h_mod = 0.88 if idx % 2 == 0 else 1.0
        curr_height = max_height * h_mod
        
        x_line = 0.12 * (t ** 3.0) * np.cos(angle)
        y_line = 0.12 * (t ** 3.0) * np.sin(angle)
        z_line = curr_height * t + 0.24 
        
        tip_x = x_line[-1]
        tip_y = y_line[-1]
        tip_z = z_line[-1]
        
        x_bead = tip_x + radius_bead * np.cos(phi) * np.sin(angle)
        y_bead = tip_y + radius_bead * np.cos(phi) * np.cos(angle)
        z_bead = tip_z + radius_bead * np.sin(phi)
        
        fig.add_trace(go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode='lines',
            line=dict(color="#fdfaa7", width=3.5), 
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter3d(
            x=x_bead, y=y_bead, z=z_bead,
            mode='lines',
            line=dict(color="#ffca6e", width=6), 
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1)

def get_sentiment_color(val):
    if val < 0.2: return "#bde0fe"   
    elif val < 0.4: return "#b8c0ff" 
    elif val < 0.6: return "#c8b6ff" 
    elif val < 0.8: return "#fecefe" 
    else: return "#fcb6cf"          

def create_hover_text(row, chapter_num, matrix_shape):
    words = int(row['word_count'])
    lexical = float(row['lexical_diversity'])
    sentiment = float(row['sentiment_polarity'])
    
    hover_string = (
        f"<b>Chapter {chapter_num}</b><br>"
        f"————————————————<br>"
        f"• Word Count: {words:,}<br>"
        f"• Lexical Diversity: {lexical:.2f}<br>"
        f"• Sentiment Polarity: {sentiment:+.2f}<br>"
    )
    return np.full(matrix_shape, hover_string)

def generate_stem_and_sepal():
    z_stem = np.linspace(-1.5, 0.0, 20)
    theta_stem = np.linspace(0, 2 * np.pi, 30)
    Z_stem, THETA_stem = np.meshgrid(z_stem, theta_stem)
    r_stem = 0.07 * (1.0 - 0.15 * Z_stem) 
    X_stem = r_stem * np.cos(THETA_stem)
    Y_stem = r_stem * np.sin(THETA_stem)
    
    u_sep = np.linspace(0, 1, 15)
    theta_sep = np.linspace(0, 2 * np.pi, 60)
    U_sep, THETA_sep = np.meshgrid(u_sep, theta_sep)
    num_points = 6
    r_sepal = 0.65 * U_sep * (1.0 + 0.25 * np.cos(num_points * THETA_sep))
    X_sepal = r_sepal * np.cos(THETA_sep)
    Y_sepal = r_sepal * np.sin(THETA_sep)
    Z_sepal = 0.25 * (U_sep ** 2) 
    
    LIFT_SEPAL_VAL = 0.05
    Z_stem += LIFT_SEPAL_VAL
    Z_sepal += LIFT_SEPAL_VAL
    
    return X_stem, Y_stem, Z_stem, X_sepal, Y_sepal, Z_sepal
# ==============================================================================
# CONFIGURATION & SUBPLOT SETUP
# ==============================================================================
BOOK_ID = "the_secret_garden"  # Options: 'alice_in_wonderland', 'jekyll_and_hyde', etc.

df = pd.read_csv(f'data/{BOOK_ID}_petals.csv')
total_chapters = len(df)
df['chapter_index'] = df.index + 1  # Standard 1-based index for clean charting axes

# --- Feature Scaling Calibrations ---
min_words, max_words = df['word_count'].min(), df['word_count'].max()
df['scaled_length'] = 1.2 + (df['word_count'] - min_words) / (max_words - min_words) * 1.0

min_lex, max_lex = df['lexical_diversity'].min(), df['lexical_diversity'].max()
df['scaled_width'] = 0.8 + (df['lexical_diversity'] - min_lex) / (max_lex - min_lex) * 0.6

min_sent, max_sent = df['sentiment_polarity'].min(), df['sentiment_polarity'].max()
df['color_val'] = (df['sentiment_polarity'] - min_sent) / (max_sent - min_sent)

max_abs_sentiment = max(abs(min_sent), abs(max_sent))
df['scaled_curl'] = -(df['sentiment_polarity'] / max_abs_sentiment) * 0.35
df['punctuation_count'] = (df['punctuation_count'] / df['word_count']) * 100
df['ripple_freq'] = 6 + (df['punctuation_count'] / df['punctuation_count'].max() * 14)

# Instantiate the Multi-Row Canvas Environment
fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.04,
    row_heights=[0.48, 0.13, 0.13, 0.13, 0.13],
    specs=[
        [{"type": "scene"}],  # Row 1: The 3D Topology Mesh Workspace
        [{"type": "xy"}],     # Row 2: Word Count Area Curve
        [{"type": "xy"}],     # Row 3: Lexical Diversity Line Matrix
        [{"type": "xy"}],     # Row 4: Sentiment Bars
        [{"type": "xy"}]      # Row 5: Punctuation Steps
    ]
)

# Add Colorbar Legend
fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None], mode='markers', # Added a dummy z axis element here
    marker=dict(
        colorscale=[[0.0, '#bde0fe'], [0.25, '#b8c0ff'], [0.5, '#c8b6ff'], [0.75, '#fecefe'], [1.0, '#fcb6cf']],
        cmin=-1.0, cmax=1.0, color=[-1.0, 1.0], showscale=True,
        colorbar=dict(
            title=dict(text="Chapter Sentiment", side="top", font=dict(size=12, family='monospace', color='#334155')),
            tickvals=[-0.8, -0.4, 0.0, 0.4, 0.8],
            ticktext=["Somber", "Negative", "Neutral", "Positive", "Whimsical"],
            thickness=15, len=0.35, x=0.94, y=0.74,
            tickfont=dict(size=10, family='monospace', color='#334155')
        )
    ),
    hoverinfo='skip', showlegend=False
), row=1, col=1)

# Render Stem & Sepal Base directly into Row 1
X_stem, Y_stem, Z_stem, X_sepal, Y_sepal, Z_sepal = generate_stem_and_sepal()
fig.add_trace(go.Surface(x=X_stem, y=Y_stem, z=Z_stem, colorscale=[[0, "#86a172"], [1, "#8cab76"]], showscale=False, hoverinfo='skip', lighting=dict(ambient=0.90, diffuse=0.50, roughness=0.7)), row=1, col=1)
fig.add_trace(go.Surface(x=X_sepal, y=Y_sepal, z=Z_sepal, colorscale=[[0, "#86a172"], [1, "#8cab76"]], showscale=False, hoverinfo='skip', lighting=dict(ambient=0.95, diffuse=0.40, roughness=0.8)), row=1, col=1)
add_stamen_cluster(fig, count=18, max_height=0.42)

# ==============================================================================
# UNIVERSAL GREEDY FIBONACCI LAYER ENGINE (Row 1)
# ==============================================================================
# Strict Fibonacci tier capacities from the inside out
FIB_SEQUENCE = [5, 8, 13, 21, 34]

# Aesthetic physics presets mapped to each prospective layer index (0=Inner, 1=Mid, 2=Outer)
TIER_PRESETS = [
    {'tilt': -0.02, 'lift_override': 0.54, 'len_mult': 1.00, 'wid_mult': 1.00, 'is_inner': True},  # Inner Core
    {'tilt': -0.08, 'lift_override': 0.28, 'len_mult': 1.00, 'wid_mult': 1.00, 'is_inner': True},  # Middle Ring
    {'tilt': -0.18, 'lift_override': 0.0,  'len_mult': 1.00, 'wid_mult': 1.00,  'is_inner': False}  # Outer Rim
]

tiers_to_render = []
chapters_allocated = 0

# Loop through our structural limits and allocate chapters greedily
for tier_idx, max_capacity in enumerate(FIB_SEQUENCE):
    if chapters_allocated >= total_chapters:
        break
        
    # Determine how many chapters are left to map
    remaining_chapters = total_chapters - chapters_allocated
    
    # Take either the full Fibonacci capacity or whatever remainder is left over
    current_tier_count = min(max_capacity, remaining_chapters)
    
    # Calculate uniform angular paths for this layer
    # Stagger odd-numbered rings slightly to allow petals to interlock naturally
    angular_offset = (np.pi / current_tier_count) if tier_idx % 2 != 0 else 0.0
    angles = np.linspace(0, 2 * np.pi, current_tier_count, endpoint=False) + angular_offset
    
    # Grab the physics presets for this specific depth layer index
    # Fall back to outer rim presets if a massive book forces an unexpected 4th tier
    preset = TIER_PRESETS[tier_idx] if tier_idx < len(TIER_PRESETS) else TIER_PRESETS[-1]
    
    # Package configuration parameters for execution
    tiers_to_render.append({
        'count': current_tier_count,
        'angles': angles,
        'is_inner': preset['is_inner'],
        'tilt': preset['tilt'],
        'lift_override': preset['lift_override'],
        'len_mult': preset['len_mult'],
        'wid_mult': preset['wid_mult'],
        'slice_start': chapters_allocated,
        'slice_end': chapters_allocated + current_tier_count
    })
    
    chapters_allocated += current_tier_count

# ==============================================================================
# UNIFIED SURFACE RENDER LIFECYCLE LOOP
# ==============================================================================
for tier_idx, tier in enumerate(tiers_to_render):
    tier_df = df.iloc[tier['slice_start']:tier['slice_end']]
    
    for idx in range(tier['count']):
        if idx >= len(tier_df):
            break
            
        row = tier_df.iloc[idx]
        is_even = (idx % 2 == 0)
        actual_chapter_num = int(row['chapter_index'])
        
        # 1. Structural Mesh Generation
        X, Y, Z = generate_highly_defined_petal(
            center_angle=tier['angles'][idx], 
            length=row['scaled_length'] * tier['len_mult'], 
            width_scale=row['scaled_width'] * tier['wid_mult'],
            tip_curl=row['scaled_curl'], 
            ripple_freq=row['ripple_freq'], 
            tilt_angle=tier['tilt'], 
            is_inner_tier=tier['is_inner'], 
            micro_lift=(True if is_even else False), 
            radial_offset=(0.14 if is_even else 0.0)
        )
        
        # 2. Physics Elevation Height Corrections
        if tier['lift_override'] is not None:
            Z = Z - (0.35 if tier['is_inner'] else 0.0) + tier['lift_override']
            
        chapter_color = get_sentiment_color(row['color_val'])
        
        # 3. Inject WebGL Tracer to Canvas
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z, 
            colorscale=[[0.0, "#ffffff"], [0.4, "#ffffff"], [0.8, chapter_color], [1.0, chapter_color]], 
            showscale=False, 
            lighting=dict(ambient=0.95, diffuse=0.40, roughness=0.8, specular=0.0),
            text=create_hover_text(row, chapter_num=actual_chapter_num, matrix_shape=X.shape), 
            hoverinfo='text'
        ), row=1, col=1)

# ==============================================================================
# SUPPLEMENTARY 2D GRAPH PLOTS GENERATION (Rows 2-5)
# ==============================================================================

# Chart 1: Word Count (Row 2)
fig.add_trace(go.Scatter(
    x=df['chapter_index'], y=df['word_count'],
    name="Word Count", fill='tozeroy',
    line=dict(color='#c8b6ff', width=2),
    fillcolor='rgba(200, 182, 255, 0.15)',
    hovertemplate="Chapter %{x}<br>Words: %{y:,}<extra></extra>"
), row=2, col=1)

# Chart 2: Lexical Diversity (Row 3)
fig.add_trace(go.Scatter(
    x=df['chapter_index'], y=df['lexical_diversity'],
    name="Lexical Diversity", mode='lines+markers',
    line=dict(color='#fcb6cf', width=2),
    marker=dict(size=5, color='#ffffff', line=dict(width=1.5, color='#fcb6cf')),
    hovertemplate="Chapter %{x}<br>TTR: %{y:.3f}<extra></extra>"
), row=3, col=1)

# Chart 3: Sentiment Polarity Diverging Bar (Row 4)
bar_colors = [get_sentiment_color(val) for val in df['color_val']]
fig.add_trace(go.Bar(
    x=df['chapter_index'], y=df['sentiment_polarity'],
    name="Sentiment", marker_color=bar_colors,
    hovertemplate="Chapter %{x}<br>Polarity: %{y:+.2f}<extra></extra>"
), row=4, col=1)

# Chart 4: Punctuation Volume Steps (Row 5)
fig.add_trace(go.Scatter(
    x=df['chapter_index'], y=df['punctuation_count'],
    name="Punctuation Velocity", mode='lines+markers',
    line=dict(color='#bde0fe', width=1.5, shape='hv', dash='dash'),
    marker=dict(size=6, symbol='diamond', color='#bde0fe', line=dict(width=1, color='#b8c0ff')),
    hovertemplate="Chapter %{x}<br>Marks: %{y}<extra></extra>"
), row=5, col=1)


# ==============================================================================
# GLOBAL UNIFIED LAYOUT TRANSFORMS
# ==============================================================================
book_title = BOOK_ID.replace("_", " ").title()

fig.update_layout(
    height=1200,  # Roomy layout height footprint for multi-tier stack navigation
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff',
    margin=dict(l=60, r=40, b=40, t=50),
    showlegend=False,
    
    # 3D Camera Scene Constants
    scene=dict(
        xaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
        zaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
        aspectratio=dict(x=1, y=1, z=0.85),
        camera=dict(eye=dict(x=1, y=1, z=1.4))
    ),
    
    # Text Titles Positioning
    annotations=[
        dict(text=book_title, font=dict(size=24, family="serif", color="#0f172a", weight="bold"), showarrow=False, x=0.02, y=0.99, xref="paper", yref="paper", xanchor="left"),
        dict(text="Narrative Informatics Blossom Topology Workspace", font=dict(size=11, family="monospace", color="#64748b"), showarrow=False, x=0.02, y=0.96, xref="paper", yref="paper", xanchor="left")
    ],
    
    hoverlabel=dict(
        bgcolor='#0f172a',      
        bordercolor='#334155',  
        font=dict(color='#f8fafc', size=12, family='monospace') 
    )
)

# Clean, minimalist axes format configuration loop across rows 2-5
for i in range(2, 6):
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1, showgrid=True, gridcolor='#f1f5f9', tickfont=dict(size=9, family='monospace'), row=i, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#cbd5e1', tickfont=dict(size=9, family='monospace'), row=i, col=1)

# Assign structural descriptive text strings onto individual Y margins
fig.update_yaxes(title_text="Word Volume", title_font=dict(size=10, family="monospace", color="#64748b"), row=2, col=1)
fig.update_yaxes(title_text="Lexical Diversity", title_font=dict(size=10, family="monospace", color="#64748b"), row=3, col=1)
fig.update_yaxes(title_text="Sentiment Arc", title_font=dict(size=10, family="monospace", color="#64748b"), row=4, col=1)
fig.update_yaxes(title_text="Punctuation Velocity", title_font=dict(size=10, family="monospace", color="#64748b"), row=5, col=1)

# Overwrite X-axis label only on the bottom chart to prevent redundancy stacking
fig.update_xaxes(title_text="Sequential Text Progress (Chapters)", title_font=dict(size=10, family="monospace", color="#64748b"), row=5, col=1)

fig.show()
fig.write_html(f"docs/assets/{BOOK_ID}_flower.html", full_html=False, include_plotlyjs='cdn')
print(f"Successfully bloomed unified workspace asset inside docs/assets/{BOOK_ID}_flower.html")
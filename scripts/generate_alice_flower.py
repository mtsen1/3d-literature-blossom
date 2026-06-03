import pandas as pd
import numpy as np
import plotly.graph_objects as go

def generate_highly_defined_petal(center_angle, length, width_scale, tip_curl, ripple_freq, tilt_angle, is_inner_tier, micro_lift, radial_offset):
    """
    Sculpts a highly chiseled teardrop data petal.
    """
    u = np.linspace(0, 1, 35)
    v = np.linspace(0, 1, 35) 
    U, V = np.meshgrid(u, v)
    
    # --- Concentric Layer Base Adjustments ---
    if is_inner_tier:
        length *= 0.82       
        width_scale *= 0.75  
        lift = 0.35          
    else:
        length *= 1.15       
        width_scale *= 1.00
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
        ))
        
        fig.add_trace(go.Scatter3d(
            x=x_bead, y=y_bead, z=z_bead,
            mode='lines',
            line=dict(color="#ffca6e", width=6), 
            showlegend=False, hoverinfo='skip'
        ))

def get_sentiment_color(val):
    if val < 0.2: return "#bde0fe"   
    elif val < 0.4: return "#b8c0ff" 
    elif val < 0.6: return "#c8b6ff" 
    elif val < 0.8: return "#fecefe" 
    else: return "#fcb6cf"          

def get_darkened_shadow_root(pastel_hex):
    return "#ffffff"

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
    
    return X_stem, Y_stem, Z_stem, X_sepal, Y_sepal, Z_sepal

# ==============================================================================
# CONFIGURATION: SWAP YOUR BOOK TOGGLE HERE
# ==============================================================================
BOOK_ID = "alice_in_wonderland"  # Options: 'alice' or 'the_secret_garden'

df = pd.read_csv(f'data/{BOOK_ID}_petals.csv')
total_chapters = len(df)

# --- Feature Scaling Calibrations ---
min_words, max_words = df['word_count'].min(), df['word_count'].max()
df['scaled_length'] = 1.2 + (df['word_count'] - min_words) / (max_words - min_words) * 1.0

min_lex, max_lex = df['lexical_diversity'].min(), df['lexical_diversity'].max()
df['scaled_width'] = 0.8 + (df['lexical_diversity'] - min_lex) / (max_lex - min_lex) * 0.6

min_sent, max_sent = df['sentiment_polarity'].min(), df['sentiment_polarity'].max()
df['color_val'] = (df['sentiment_polarity'] - min_sent) / (max_sent - min_sent)

max_abs_sentiment = max(abs(min_sent), abs(max_sent))
df['scaled_curl'] = -(df['sentiment_polarity'] / max_abs_sentiment) * 0.35
df['ripple_freq'] = 6 + (df['punctuation_count'] / df['punctuation_count'].max() * 14)

# Initialize staging grounds
fig = go.Figure()

# Add Colorbar Legend
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode='markers',
    marker=dict(
        colorscale=[[0.0, '#bde0fe'], [0.25, '#b8c0ff'], [0.5, '#c8b6ff'], [0.75, '#fecefe'], [1.0, '#fcb6cf']],
        cmin=-1.0, cmax=1.0, color=[-1.0, 1.0], showscale=True,
        colorbar=dict(
            title=dict(text="Chapter Sentiment", side="top", font=dict(size=12, family='monospace', color='#334155')),
            tickvals=[-0.8, -0.4, 0.0, 0.4, 0.8],
            ticktext=["Somber", "Negative", "Neutral", "Positive", "Whimsical"],
            thickness=16, len=0.45, x=0.92, y=0.5,
            tickfont=dict(size=11, family='monospace', color='#334155')
        )
    ),
    hoverinfo='skip', showlegend=False
))

# Render Stem & Sepal Base
X_stem, Y_stem, Z_stem, X_sepal, Y_sepal, Z_sepal = generate_stem_and_sepal()
fig.add_trace(go.Surface(x=X_stem, y=Y_stem, z=Z_stem, colorscale=[[0, "#86a172"], [1, "#8cab76"]], showscale=False, hoverinfo='skip', lighting=dict(ambient=0.90, diffuse=0.50, roughness=0.7)))
fig.add_trace(go.Surface(x=X_sepal, y=Y_sepal, z=Z_sepal, colorscale=[[0, "#86a172"], [1, "#8cab76"]], showscale=False, hoverinfo='skip', lighting=dict(ambient=0.95, diffuse=0.40, roughness=0.8)))
add_stamen_cluster(fig, count=18, max_height=0.42)

# ==============================================================================
# DYNAMIC CONCENTRIC TIER DISTRIBUTION ENGINE
# ==============================================================================
if total_chapters < 12:
    # Small texts fit entirely on one uniform ring
    outer_count = total_chapters
    inner_count = 0
else:
    # Large texts are split evenly down the middle into dual concentric rings
    outer_count = total_chapters // 2
    inner_count = total_chapters - outer_count

angles_outer = np.linspace(0, 2 * np.pi, outer_count, endpoint=False)
angles_inner = np.linspace(0, 2 * np.pi, inner_count, endpoint=False) + (np.pi / max(inner_count, 1))

# --- LOOP 1: RENDER OUTER TIER ---
for idx in range(outer_count):
    row = df.iloc[idx]
    is_even = (idx % 2 == 0)
    X, Y, Z = generate_highly_defined_petal(
        center_angle=angles_outer[idx], length=row['scaled_length'], width_scale=row['scaled_width'],
        tip_curl=row['scaled_curl'], ripple_freq=row['ripple_freq'], tilt_angle=-0.16, 
        is_inner_tier=False, micro_lift=(True if is_even else False), radial_offset=(0.15 if is_even else 0.0)
    )
    chapter_color = get_sentiment_color(row['color_val'])
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, colorscale=[[0.0, "#ffffff"], [0.4, "#ffffff"], [0.8, chapter_color], [1.0, chapter_color]], 
        showscale=False, lighting=dict(ambient=0.95, diffuse=0.40, roughness=0.8, specular=0.0),
        text=create_hover_text(row, chapter_num=idx+1, matrix_shape=X.shape), hoverinfo='text'
    ))

# --- LOOP 2: RENDER INNER TIER ---
for idx in range(inner_count):
    row = df.iloc[idx + outer_count]
    is_even = (idx % 2 == 0)
    X, Y, Z = generate_highly_defined_petal(
        center_angle=angles_inner[idx], length=row['scaled_length'], width_scale=row['scaled_width'],
        tip_curl=row['scaled_curl'], ripple_freq=row['ripple_freq'], tilt_angle=-0.04, 
        is_inner_tier=True, micro_lift=(True if is_even else False), radial_offset=(0.12 if is_even else 0.0)
    )
    chapter_color = get_sentiment_color(row['color_val'])
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, colorscale=[[0.0, "#ffffff"], [0.4, "#ffffff"], [0.8, chapter_color], [1.0, chapter_color]], 
        showscale=False, lighting=dict(ambient=0.95, diffuse=0.40, roughness=0.8, specular=0.0),
        text=create_hover_text(row, chapter_num=idx+outer_count+1, matrix_shape=X.shape), hoverinfo='text'
    ))

# Layout configurations
book_title = BOOK_ID.replace("_", " ").title()
# Viewport formatting & Absolute Minimalist Grid Elimination
fig.update_layout(
    scene=dict(
        # Explicitly turn off visibility, background panels, and grid grids for all three dimensions
        xaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
        zaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
        aspectratio=dict(x=1, y=1, z=0.95),
        camera=dict(
            eye=dict(x=1.25, y=1.25, z=0.8)
        )
    ),
    annotations=[
        dict(text=book_title, font=dict(size=26, family="serif", color="#0f172a", weight="bold"), showarrow=False, x=0.04, y=0.94, xref="paper", yref="paper", xanchor="left"),
        dict(text="Narrative Informatics Blossom Topology", font=dict(size=12, family="monospace", color="#64748b"), showarrow=False, x=0.04, y=0.89, xref="paper", yref="paper", xanchor="left")
    ],
    hoverlabel=dict(
        bgcolor='#0f172a',      
        bordercolor='#334155',  
        font=dict(color='#f8fafc', size=13, family='monospace') 
    ),
    paper_bgcolor='rgba(255,255,255,1)',
    plot_bgcolor='rgba(255,255,255,1)',
    margin=dict(l=40, r=40, b=20, t=20),
    showlegend=False
)

fig.show()
fig.write_html(f"docs/assets/{BOOK_ID}_flower.html", full_html=False, include_plotlyjs='cdn')
print(f"Successfully bloomed dynamic asset inside docs/assets/{BOOK_ID}_flower.html")
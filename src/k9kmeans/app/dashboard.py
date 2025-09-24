import argparse
import subprocess
import sys
import os
import pandas as pd
from PIL import Image
import streamlit as st

# Number of images per row in the dashboard
images_per_row = 5


def run_dashboard(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    st.title('Dog Image Clusters 🐶')
    st.write('Select a cluster')

    # Dropdown menu for clusters
    clusters = sorted(df['cluster'].unique())
    selected_cluster = st.selectbox('Choose a cluster', clusters)

    cluster_df = df[df['cluster'] == selected_cluster]
    st.write(f'Cluster {selected_cluster} contains {len(cluster_df)} images.')

    # Display images in grid
    rows = [
        cluster_df['filename'].tolist()[i : i + images_per_row]
        for i in range(0, len(cluster_df), images_per_row)
    ]

    for row in rows:
        cols = st.columns(len(row))
        for col, filename in zip(cols, row):
            try:
                img = Image.open(filename)
                col.image(img, use_container_width=True, width='stretch')
            except Exception:
                col.error(f'Error loading {filename}')


def cli() -> None:
    parser = argparse.ArgumentParser(description="Launch the dog image dashboard")
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file with filename and cluster columns',
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        os.path.abspath(__file__),
        '--',
        f'--csv={args.csv}',
    ]

    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    import sys

    if 'streamlit' in sys.modules:
        # Running inside streamlit, parse Streamlit arguments
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--csv', type=str, required=True)
        args, _ = parser.parse_known_args()
        run_dashboard(csv_path=args.csv)
    else:
        cli()

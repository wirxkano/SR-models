import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_plot_across_conditions(path: str, dataset: str, metric: str):
    df_opencv = pd.read_excel(path, sheet_name="Bicubic_OpenCV", header=[0,1])

    df_opencv.columns = [
        col if not isinstance(col, tuple) else col
        for col in df_opencv.columns
    ]

    df_opencv = df_opencv.rename(columns={
        ('Solution Name', 'Unnamed: 0_level_1'): 'Method',
        ('Evaluation Conditions', 'Unnamed: 1_level_1'): 'Condition'
    })

    df_matlab = pd.read_excel(path, sheet_name="Bicubic_MATLAB", header=[0,1])

    df_matlab.columns = [
        col if not isinstance(col, tuple) else col
        for col in df_matlab.columns
    ]

    df_matlab = df_matlab.rename(columns={
        ('Solution Name', 'Unnamed: 0_level_1'): 'Method',
        ('Evaluation Conditions', 'Unnamed: 1_level_1'): 'Condition'
    })

    df_iso = pd.read_excel(path, sheet_name="Isotropic", header=[0,1])

    df_iso.columns = [
        col if not isinstance(col, tuple) else col
        for col in df_iso.columns
    ]

    df_iso = df_iso.rename(columns={
        ('Solution Name', 'Unnamed: 0_level_1'): 'Method',
        ('Evaluation Conditions', 'Unnamed: 1_level_1'): 'Condition'
    })

    df_aniso = pd.read_excel(path, sheet_name="Anisotropic", header=[0,1])

    df_aniso.columns = [
        col if not isinstance(col, tuple) else col
        for col in df_aniso.columns
    ]

    df_aniso = df_aniso.rename(columns={
        ('Solution Name', 'Unnamed: 0_level_1'): 'Method',
        ('Evaluation Conditions', 'Unnamed: 1_level_1'): 'Condition'
    })

    # dataset = "Urban100"
    # metric = "PSNR"

    methods = df_opencv['Method']
    values_opencv = df_opencv[(dataset, metric)]
    values_matlab = df_matlab[(dataset, metric)]
    values_iso = df_iso[(dataset, metric)]
    values_aniso = df_aniso[(dataset, metric)]

    mask = values_opencv.notna()
    methods = methods[mask]
    values_opencv = values_opencv[mask]
    values_matlab = values_matlab[mask]
    values_iso = values_iso[mask]
    values_aniso = values_aniso[mask]


    x = np.arange(len(methods))

    plt.figure(figsize=(12, 5))
    plt.plot(x, values_opencv, marker='o', linewidth=2, label='OpenCV')
    plt.plot(x, values_matlab, marker='o', linewidth=2, label='MATLAB')
    plt.plot(x, values_iso, marker='o', linewidth=2, label='Isotropic')
    plt.plot(x, values_aniso, marker='o', linewidth=2, label='Anisotropic')

    plt.xticks(x, methods, rotation=30, ha='right')
    plt.xlabel("Method")
    plt.ylabel("PSNR (dB)")
    plt.title(f"PSNR on {dataset}")

    plt.grid(True, linestyle='--', alpha=0.4)

    plt.legend(ncol=3)
    plt.tight_layout()
    plt.show()

def draw_plot_across_datasets(path: str, sheet_name: str, datasets: list, metric: str):
    df = pd.read_excel(path, sheet_name=sheet_name, header=[0, 1])
    
    df = df.rename(columns={
        'Unnamed: 0_level_0': 'Metadata',
        'Unnamed: 1_level_0': 'Metadata'
    })

    methods = df[('Solution Name', 'Unnamed: 0_level_1')]
    
    results = {}
    for idx, method in enumerate(methods):
        method_results = []
        for dataset in datasets:
            val = df.loc[idx, (dataset, metric)]
            method_results.append(val)
        results[method] = method_results

    plt.figure(figsize=(10, 6))
    for method, psnr_values in results.items():
        plt.plot(
            datasets,
            psnr_values,
            marker='o',
            linewidth=2,
            label=method
        )

    plt.xlabel("Dataset")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR across datasets")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend(
        loc='best',
        fontsize=9,
        ncol=2
    )

    plt.tight_layout()
    plt.show()
    
def draw_bar_chart_flops():
    methods = ["DAN", "DASR", "MANet", "Real-ESRGAN", "BSRGAN", "KOALAnet", "DCLS", "CMOS", "ASBSR"]
    gflops = [2500.74, 421.82, 1198.40, 2350.48, 2350.48, 25.90, 993.58, 2513.86, 1864.59]
    
    plt.bar(methods, gflops)
    plt.xlabel("Method")
    plt.ylabel("GFLOPS")
    plt.title("GFLOPS Comparison")
    
    plt.show()
    
def draw_line_chart_exetime():
    methods = ["DAN", "DASR", "MANet", "Real-ESRGAN", "BSRGAN", "KOALAnet", "DCLS", "CMOS", "ASBSR"]
    time_2080 = [366.16, 86.11, 176.64, 223.39, 288.05, 213.49, 176.80, 477.25, 1347.41]
    time_p100 = [351.53, 85.20, 164.32, 334.98, 263.99, 205.35, 185.16, 552.31, 1344.22]
    time_a100 = [146.79, 57.20, 63.03, 138.55, 126.79, 62.65, 433.26, 531.79, 518.65]
    
    time_dict = {
        "2080 Super": time_2080,
        "P100": time_p100,
        "A100": time_a100
    }
    
    markers = ['o', 's', '^']
    
    for i, (name, exetime) in enumerate(time_dict.items()):
        plt.plot(methods, exetime, linewidth=2, marker=markers[i], label=name)
        
    plt.xlabel("Method")
    plt.ylabel("Execution Time (ms)")
    plt.title("Execution Time Comparison")
    plt.xticks(rotation=45, ha="right")
    
    plt.legend(title="GPU")
    
    plt.show()
    
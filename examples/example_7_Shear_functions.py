def plot_shear_profile(winds, levs, title):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(winds, levs, marker='o')
    plt.xlabel('Mean Wind Speed (m/s)')
    plt.ylabel('height (m)')
    plt.title(f'Mean Wind Speed Profile {title.capitalize()}')
    plt.grid()
    plt.show()

#########plot_shear_profile(ds_vortex_mean, ds_vortex['lev'].values, SITE)
#

def plot_shear_profile_by(_ds, levs, grouping, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    groupings = _ds[grouping].values
    print(groupings)
    for i, sel in enumerate(groupings):
        print(f'Plotting {grouping} {sel}')
        winds = _ds.sel({grouping: sel})['M'].values.flatten()
        plt.plot(winds, levs, marker='o', label=f'{sel:02d}')
    plt.xlabel('Mean Wind Speed (m/s)')
    plt.ylabel('Height (m)')
    plt.title(f'Mean Wind Speed Profile by {grouping.capitalize()} - {title.capitalize()}')
    plt.grid()
    plt.legend(title=grouping.capitalize())
    # show this in the graph to see the stability classes
        # 0- Very Unstable: RI < -0.05
        #1 - Unstable: -0.05 <= RI < 0
        #2 - Near-neutral Unstable: 0 <= RI < 0.01
        #3 - Neutral: 0.01 <= RI < 0.05
        #4- Near-neutral Stable: 0.05 <= RI < 0.25
        #5 - Stable: 0.25 <= RI < 1.0
        #6- Very Stable: RI >= 1. 
    # Add a text box with stability class definitions
    if grouping == "stability":
            
        stability_text = (
            "Stability Classes:\n"
            "0 - Very Unstable: RI < -0.05\n"
            "1 - Unstable: -0.05 <= RI < 0\n"
            "2 - Near-neutral Unstable: 0 <= RI < 0.01\n"
            "3 - Neutral: 0.01 <= RI < 0.05\n"
            "4 - Near-neutral Stable: 0.05 <= RI < 0.25\n"
            "5 - Stable: 0.25 <= RI < 1.0\n"
            "6 - Very Stable: RI >= 1.0"
        )
        plt.gcf().text(0.15, 0.02, stability_text, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.show()
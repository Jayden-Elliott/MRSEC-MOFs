import os
import sys
import pandas as pd
import numpy as np
import subprocess
from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive, get_MOF_descriptors

PATHNAME = os.path.abspath('.') + '/'
ZEO_PATHNAME = "/Users/jayden/Documents/Code/MOFS_MRSEC/packages/zeo++-0.3/"


def main(folder):
    if not os.path.exists(folder + "zeo/"):
        os.mkdir(folder + "zeo/")
    zeo_folder = folder + "zeo/"
    cif_folder = folder + "cifs/"

    featurization_list = []
    for filename in os.listdir(folder + "cifs/"):
        try:
            if not os.path.exists(folder + "primitives/" + filename):
                get_primitive(folder + "cifs/" + filename,
                              folder + "primitives/" + filename)
            name = filename[:-4]

            cmd1 = ZEO_PATHNAME + "network -ha -res " + \
                zeo_folder + name + "_pd.txt " + cif_folder + filename
            cmd2 = ZEO_PATHNAME + "network -sa 1.86 1.86 10000 " + \
                ZEO_PATHNAME + "_sa.txt " + cif_folder + filename
            cmd3 = PATHNAME + "zeo++-0.3/network -volpo 1.86 1.86 10000 " + \
                zeo_folder + name + "_pov.txt " + cif_folder + filename

            process1 = subprocess.Popen(
                cmd1, stdout=subprocess.PIPE, stderr=None, shell=True)
            process2 = subprocess.Popen(
                cmd2, stdout=subprocess.PIPE, stderr=None, shell=True)
            process3 = subprocess.Popen(
                cmd3, stdout=subprocess.PIPE, stderr=None, shell=True)

            output1 = process1.communicate()[0]
            output2 = process2.communicate()[0]
            output3 = process3.communicate()[0]

            cif_file = name + "_primitive.cif"
            basename = cif_file.strip(".cif")
            largest_included_sphere, largest_free_sphere, largest_included_sphere_along_free_sphere_path = np.nan, np.nan, np.nan
            unit_cell_volume, crystal_density, VSA, GSA = np.nan, np.nan, np.nan, np.nan
            VPOV, GPOV = np.nan, np.nan
            POAV, PONAV, GPOAV, GPONAV, POAV_volume_fraction, PONAV_volume_fraction = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            if (os.path.exists(zeo_folder + name + "_pd.txt") & os.path.exists(zeo_folder + name + "_sa.txt") & os.path.exists(zeo_folder + name + "_pov.txt")):
                with open(zeo_folder + name + "_pd.txt") as f:
                    pore_diameter_data = f.readlines()
                    for row in pore_diameter_data:
                        largest_included_sphere = float(
                            row.split()[1])  # largest included sphere
                        largest_free_sphere = float(
                            row.split()[2])  # largest free sphere
                        largest_included_sphere_along_free_sphere_path = float(
                            row.split()[3])  # largest included sphere along free sphere path
                with open(zeo_folder + name + "_sa.txt") as f:
                    surface_area_data = f.readlines()
                    for i, row in enumerate(surface_area_data):
                        if i == 0:
                            unit_cell_volume = float(row.split("Unitcell_volume:")[
                                1].split()[0])  # unit cell volume
                            crystal_density = float(row.split("Unitcell_volume:")[
                                                    1].split()[0])  # crystal density
                            # volumetric surface area
                            VSA = float(row.split("ASA_m^2/cm^3:")
                                        [1].split()[0])
                            # gravimetric surface area
                            GSA = float(row.split("ASA_m^2/g:")[1].split()[0])
                with open(zeo_folder + name + "_pov.txt") as f:
                    pore_volume_data = f.readlines()
                    for i, row in enumerate(pore_volume_data):
                        if i == 0:
                            density = float(
                                row.split("Density:")[1].split()[0])
                            # Probe accessible pore volume
                            POAV = float(row.split("POAV_A^3:")[1].split()[0])
                            # Probe non-accessible probe volume
                            PONAV = float(
                                row.split("PONAV_A^3:")[1].split()[0])
                            GPOAV = float(
                                row.split("POAV_cm^3/g:")[1].split()[0])
                            GPONAV = float(
                                row.split("PONAV_cm^3/g:")[1].split()[0])
                            POAV_volume_fraction = float(row.split("POAV_Volume_fraction:")[1].split()[
                                0])  # probe accessible volume fraction
                            PONAV_volume_fraction = float(row.split("PONAV_Volume_fraction:")[1].split()[
                                0])  # probe non accessible volume fraction
                            VPOV = POAV_volume_fraction + PONAV_volume_fraction
                            GPOV = VPOV / density
            else:
                print("Not all 3 files exist, so at least one Zeo++ call failed!", "sa: ", os.path.exists(zeo_folder + name + "_sa.txt"),
                      "; pd: ", os.path.exists(zeo_folder + name + "_pd.txt"), "; pov: ", os.path.exists(zeo_folder + name + "_pov.txt"))
                continue
            featurization = {"name": basename, "cif_file": cif_file, "Di": largest_included_sphere, "Df": largest_free_sphere, "Dif": largest_included_sphere_along_free_sphere_path,
                             "rho": crystal_density, "VSA": VSA, "GSA": GSA, "VPOV": VPOV, "GPOV": GPOV, "POAV_vol_frac": POAV_volume_fraction,
                             "PONAV_vol_frac": PONAV_volume_fraction, "GPOAV": GPOAV, "GPONAV": GPONAV, "POAV": POAV, "PONAV": PONAV}
            featurization_list.append(featurization)
        except:
            continue
    df = pd.DataFrame(featurization_list)
    df.to_csv(folder + "geo_features.csv", index=False)
    return df.sort_values(by=["name"])


if __name__ == '__main__':
    folder = sys.argv[1]
    main(PATHNAME + folder + '/')

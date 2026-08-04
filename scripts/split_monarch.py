import xarray as xr
from pathlib import Path
from datetime import datetime

# ---------------------
# User parameters
# ---------------------
input_dir = Path("/home/tvintimi/data/providentia/exp_to_interp/a00c_cirrus/regional/3hourly")   # <-- Change to your input directory
output_dir = Path("/home/tvintimi/data/providentia/exp_to_interp/a00c_cirrus/regional/3hourly") # <-- Change to your output directory
start_date = "20250302"  # <-- YYYYMMDD
end_date   = "20250307"  # <-- YYYYMMDD

# Variables to extract
var_groups = {
    "od550_dust": ["lat", "lon", "time", "od550_dust"],
    #"dust_load": ["lat", "lon", "time", "dust_load"],
    "sconc_dust": ["lat", "lon", "time", "sconc_dust"]
}

# ---------------------
# Prep
# ---------------------
start_dt = datetime.strptime(start_date, "%Y%m%d")
end_dt   = datetime.strptime(end_date, "%Y%m%d")

# ---------------------
# Process files
# ---------------------
for f in sorted(input_dir.glob("*.nc")):
    try:
        # Extract datetime from filename (assume format YYYYMMDD12_...)
        date_str_full = f.stem.split("_")[0]  # e.g., 2020010112
        date_str = date_str_full[:8]           # YYYYMMDD
        file_date = datetime.strptime(date_str, "%Y%m%d")
    except Exception:
        continue  # skip files not matching pattern

    if not (start_dt <= file_date <= end_dt):
        continue  # skip outside range

    print(f"Processing {f.name}...")
    ds = xr.open_dataset(f)

    for v, selection in var_groups.items():
        subset = ds[selection]

        # Avoid FillValue conflicts
        enc = {}
        for var in subset.data_vars:
            enc[var] = {"_FillValue": None}

        # Output filename: var_YYYYMMDD12.nc (using first 10 chars of input filename)
        out_file = output_dir / v / f"{v}_{date_str_full}.nc"
        subset.to_netcdf(out_file, encoding=enc)

print(f"✅ Processing complete. Files written to {output_dir}")

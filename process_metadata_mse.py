import argparse
import csv
from pathlib import Path


X_COL = "valve_x_m"
Y_COL = "valve_y_m"
Z_COL = "valve_z_m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute 3D MSE and axis means from a metadata CSV."
    )
    parser.add_argument(
        "--input",
        default="metadata.csv",
        help="Input CSV path. Default: metadata.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: overwrite input CSV",
    )
    parser.add_argument("--true-x", type=float, default=0.3, help="Ground-truth X")
    parser.add_argument("--true-y", type=float, default=-0.15, help="Ground-truth Y")
    parser.add_argument("--true-z", type=float, default=0.5, help="Ground-truth Z")
    return parser.parse_args()


def read_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        rows = list(reader)
        return reader.fieldnames, rows


def process_rows(rows, true_x, true_y, true_z):
    if not rows:
        raise ValueError("CSV contains no data rows.")

    x_values = []
    y_values = []
    z_values = []
    total_squared_error = 0.0

    for row in rows:
        try:
            x_val = float(row[X_COL])
            y_val = float(row[Y_COL])
            z_val = float(row[Z_COL])
        except KeyError as exc:
            raise KeyError(
                f"Missing required column: {exc.args[0]}. "
                f"Expected columns: {X_COL}, {Y_COL}, {Z_COL}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value found: {exc}") from exc

        x_values.append(x_val)
        y_values.append(y_val)
        z_values.append(z_val)

        x_error_sq = (x_val - true_x) ** 2
        y_error_sq = (y_val - true_y) ** 2
        z_error_sq = (z_val - true_z) ** 2
        point_se = x_error_sq + y_error_sq + z_error_sq
        total_squared_error += point_se

        row["x_error_sq"] = f"{x_error_sq:.12f}"
        row["y_error_sq"] = f"{y_error_sq:.12f}"
        row["z_error_sq"] = f"{z_error_sq:.12f}"
        row["point_se_3d"] = f"{point_se:.12f}"
        row["mse_3d"] = ""
        row["mean_x"] = ""
        row["mean_y"] = ""
        row["mean_z"] = ""
        row["true_x"] = f"{true_x:.12f}"
        row["true_y"] = f"{true_y:.12f}"
        row["true_z"] = f"{true_z:.12f}"
        row["summary_type"] = ""

    count = len(rows)
    mse_3d = total_squared_error / count
    mean_x = sum(x_values) / count
    mean_y = sum(y_values) / count
    mean_z = sum(z_values) / count

    summary_row = {key: "" for key in rows[0].keys()}
    summary_row["frame_index"] = "SUMMARY"
    summary_row["summary_type"] = "dataset_metrics"
    summary_row["mse_3d"] = f"{mse_3d:.12f}"
    summary_row["mean_x"] = f"{mean_x:.12f}"
    summary_row["mean_y"] = f"{mean_y:.12f}"
    summary_row["mean_z"] = f"{mean_z:.12f}"
    summary_row["true_x"] = f"{true_x:.12f}"
    summary_row["true_y"] = f"{true_y:.12f}"
    summary_row["true_z"] = f"{true_z:.12f}"

    rows.append(summary_row)
    return rows, mse_3d, mean_x, mean_y, mean_z


def write_rows(csv_path: Path, fieldnames, rows):
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    fieldnames, rows = read_rows(input_path)
    processed_rows, mse_3d, mean_x, mean_y, mean_z = process_rows(
        rows, args.true_x, args.true_y, args.true_z
    )

    extra_columns = [
        "x_error_sq",
        "y_error_sq",
        "z_error_sq",
        "point_se_3d",
        "mse_3d",
        "mean_x",
        "mean_y",
        "mean_z",
        "true_x",
        "true_y",
        "true_z",
        "summary_type",
    ]
    final_fieldnames = fieldnames + [col for col in extra_columns if col not in fieldnames]
    write_rows(output_path, final_fieldnames, processed_rows)

    print(f"Processed CSV written to: {output_path}")
    print(f"3D MSE  : {mse_3d:.12f}")
    print(f"Mean X  : {mean_x:.12f}")
    print(f"Mean Y  : {mean_y:.12f}")
    print(f"Mean Z  : {mean_z:.12f}")


if __name__ == "__main__":
    main()

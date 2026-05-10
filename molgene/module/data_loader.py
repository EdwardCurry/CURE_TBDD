import anndata as ad
import numpy as np
import pandas as pd
import sys
import os
import glob
import time
import multiprocessing # For parallel processing
import logging # For better error logging

# --- Configuration ---
# Input directory containing .h5ad files
INPUT_DIR = './projects_list/arc-virtual-cell-atlas/sc_embed/'
# Output directory for CSV files (can be the same or different)
OUTPUT_DIR = './' # Save to current directory, change if needed
LOG_FILE = 'pseudo_cell_generation.log' # File to log errors and progress

# Parameters for pseudo-cell generation
TOTAL_PSEUDO_CELLS = 128
N_CELLS_PER_PSEUDO = 15
NUM_REPETITIONS = 5 # Number of times to repeat the generation for each file

# Parallel processing settings
# Use slightly less than the total number of CPUs to leave resources for the OS
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 2)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout) # Also print logs to console
    ]
)

# --- Core Processing Function for a Single H5AD File ---
def process_h5ad_file(h5ad_path):
    """
    Reads a single .h5ad file, generates pseudo-cells 5 times,
    and saves the results to CSV files.
    Designed to be run in a separate process.
    """
    base_name = os.path.splitext(os.path.basename(h5ad_path))[0]
    output_csv_base_name = os.path.join(OUTPUT_DIR, base_name)
    process_id = os.getpid() # Get process ID for logging
    logging.info(f"[PID {process_id}] Processing file: {h5ad_path}")

    try:
        # --- 1. Read Input AnnData File ---
        try:
            adata = ad.read_h5ad(h5ad_path)
            logging.debug(f"[PID {process_id}] File read successfully: {h5ad_path}")
            # Basic check for required data
            if 'phase' not in adata.obs.columns:
                logging.error(f"[PID {process_id}] 'phase' column not found in {h5ad_path}. Skipping.")
                return f"Skipped: Missing 'phase' column in {h5ad_path}"
        except FileNotFoundError:
            logging.error(f"[PID {process_id}] Input file not found: {h5ad_path}. Skipping.")
            return f"Skipped: File not found {h5ad_path}"
        except Exception as e:
            logging.error(f"[PID {process_id}] Error reading H5AD file {h5ad_path}: {e}. Skipping.")
            return f"Skipped: Error reading H5AD {h5ad_path} - {e}"

        # --- 2. Calculate Phase Proportions and Target Counts ---
        try:
            phase_counts = adata.obs['phase'].value_counts()
            total_real_cells = phase_counts.sum()
            if total_real_cells == 0:
                logging.warning(f"[PID {process_id}] No cells found in {h5ad_path}. Skipping.")
                return f"Skipped: No cells in {h5ad_path}"

            proportions = phase_counts / total_real_cells
            target_counts_float = proportions * TOTAL_PSEUDO_CELLS
            target_counts = target_counts_float.round().astype(int)
            diff = TOTAL_PSEUDO_CELLS - target_counts.sum()

            if diff != 0:
                if not target_counts.empty:
                    idx_to_adjust = target_counts.idxmax()
                    target_counts[idx_to_adjust] += diff
                else:
                     logging.warning(f"[PID {process_id}] Cannot adjust counts as target_counts is empty for {h5ad_path}. Total might not be {TOTAL_PSEUDO_CELLS}.")


            # Final check (allow processing even if assertion fails, but log it)
            if target_counts.sum() != TOTAL_PSEUDO_CELLS:
                 logging.warning(f"[PID {process_id}] Calculated pseudo-cell total {target_counts.sum()} != target {TOTAL_PSEUDO_CELLS} for {h5ad_path}")

            logging.debug(f"[PID {process_id}] Target counts calculated for {h5ad_path}:\n{target_counts}")

        except Exception as e:
            logging.error(f"[PID {process_id}] Error during count calculation for {h5ad_path}: {e}. Skipping.")
            return f"Skipped: Count calculation error {h5ad_path} - {e}"

        # --- Loop for Repetitions ---
        files_generated_count = 0
        for n in range(1, NUM_REPETITIONS + 1):
            logging.debug(f"[PID {process_id}] Starting repetition {n} for {h5ad_path}")
            pseudo_cell_list = []
            pseudo_cell_phases = []
            generation_successful_rep = True

            # --- 3. Generate Pseudo-cells (Sampling/Averaging) ---
            try:
                for phase in target_counts.index:
                    num_pseudo_for_phase = target_counts.get(phase, 0) # Use .get for safety
                    if num_pseudo_for_phase <= 0:
                        continue

                    real_cell_indices_phase = adata.obs.index[adata.obs['phase'] == phase]
                    n_real_cells_in_phase = len(real_cell_indices_phase)

                    if n_real_cells_in_phase < N_CELLS_PER_PSEUDO:
                        logging.warning(f"[PID {process_id}] Phase '{phase}' in {h5ad_path} has only {n_real_cells_in_phase} cells (< {N_CELLS_PER_PSEUDO}). Skipping phase for repetition {n}.")
                        continue

                    for i in range(num_pseudo_for_phase):
                        chosen_indices = np.random.choice(real_cell_indices_phase,
                                                          size=N_CELLS_PER_PSEUDO,
                                                          replace=False)
                        pseudo_expression = adata[chosen_indices, :].X.mean(axis=0)

                        if hasattr(pseudo_expression, "A"):
                            pseudo_expression = pseudo_expression.A.flatten()
                        elif isinstance(pseudo_expression, np.matrix):
                            pseudo_expression = np.array(pseudo_expression).flatten()
                        elif not isinstance(pseudo_expression, np.ndarray):
                             pseudo_expression = np.array(pseudo_expression)
                        if pseudo_expression.ndim > 1:
                            pseudo_expression = pseudo_expression.flatten()

                        pseudo_cell_list.append(pseudo_expression)
                        pseudo_cell_phases.append(phase)

            except Exception as e:
                logging.error(f"[PID {process_id}] Error during generation in repetition {n} for {h5ad_path}: {e}")
                generation_successful_rep = False

            # --- 4. Prepare Data for Saving ---
            if not generation_successful_rep or not pseudo_cell_list:
                logging.warning(f"[PID {process_id}] No pseudo-cells generated in repetition {n} for {h5ad_path}. Skipping save.")
                continue

            try:
                pseudo_cell_matrix = np.vstack(pseudo_cell_list)
                n_generated = pseudo_cell_matrix.shape[0]
                if n_generated != len(pseudo_cell_phases):
                     logging.warning(f"[PID {process_id}] Mismatch rows/labels ({n_generated}/{len(pseudo_cell_phases)}) in rep {n} for {h5ad_path}")

                pseudo_df = pd.DataFrame(pseudo_cell_matrix, columns=adata.var_names)
                pseudo_df.insert(0, 'phase', pseudo_cell_phases[:n_generated])
                pseudo_df.index = [f"pseudo_{i+1}" for i in range(n_generated)]
                pseudo_df.index.name = "PseudoCellID"

            except Exception as e:
                logging.error(f"[PID {process_id}] Error preparing data in repetition {n} for {h5ad_path}: {e}")
                continue

            # --- 5. Save Data to CSV ---
            output_csv_path = f"{output_csv_base_name}+{n}.csv"
            try:
                pseudo_df.to_csv(output_csv_path, index=True)
                logging.debug(f"[PID {process_id}] Data for repetition {n} saved to: {output_csv_path}")
                files_generated_count += 1
            except Exception as e:
                logging.error(f"[PID {process_id}] Error saving CSV {output_csv_path}: {e}")

        # --- End Repetition Loop ---
        logging.info(f"[PID {process_id}] Finished processing {h5ad_path}. Generated {files_generated_count}/{NUM_REPETITIONS} CSV files.")
        return f"Success: {h5ad_path} - Generated {files_generated_count} files."

    except Exception as e:
        # Catch any unexpected errors during the whole file processing
        logging.exception(f"[PID {process_id}] Unexpected error processing file {h5ad_path}: {e}")
        return f"Failed: Unexpected error in {h5ad_path} - {e}"

# --- Main Execution ---
if __name__ == '__main__':
    logging.info("="*50)
    logging.info(f"Script started at: {pd.Timestamp.now()}")
    logging.info(f"Input Directory: {INPUT_DIR}")
    logging.info(f"Output Directory: {OUTPUT_DIR}")
    logging.info(f"Number of Repetitions per file: {NUM_REPETITIONS}")
    logging.info(f"Number of Worker Processes: {NUM_PROCESSES}")
    logging.info(f"Log File: {LOG_FILE}")
    logging.info("="*50)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all .h5ad files in the input directory
    h5ad_files = glob.glob(os.path.join(INPUT_DIR, '*.h5ad'))
    num_files = len(h5ad_files)
    logging.info(f"Found {num_files} .h5ad files to process.")

    if num_files == 0:
        logging.warning("No .h5ad files found in the input directory. Exiting.")
        sys.exit(0)

    start_time = time.time()

    # Create a pool of worker processes
    logging.info(f"Creating multiprocessing pool with {NUM_PROCESSES} workers...")
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        # Use imap_unordered for potentially better performance with many tasks
        # It yields results as they complete, not necessarily in order
        results = list(pool.imap_unordered(process_h5ad_file, h5ad_files))
        # Use map if order matters or for simpler result handling:
        # results = pool.map(process_h5ad_file, h5ad_files)

    end_time = time.time()
    total_time = end_time - start_time

    logging.info("="*50)
    logging.info(f"Multiprocessing finished.")
    logging.info(f"Total files processed (attempted): {num_files}")
    # You could analyze 'results' list here to count successes/failures if needed
    # Example: success_count = sum(1 for r in results if r.startswith("Success"))
    # logging.info(f"Successful file processing count: {success_count}")
    logging.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logging.info(f"Script finished successfully at: {pd.Timestamp.now()}")
    logging.info("="*50)
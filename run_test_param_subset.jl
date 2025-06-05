using .SimulationSpace, .LoadData
using JLD2

# Read job index (1-based SLURM_ARRAY_TASK_ID)
job_id = parse(Int, ARGS[1])

#load stuff
rad_sim = LoadData.import_data(joinpath("sim_data", "meshtap"))
robot_starts = SimulationSpace.gen_sample_starts(num_samples=15)

#set up param sweeps
exploring_start_steps = [20, 17, 15, 12, 10, 5]
r_check_vals = [100.0, 75.0, 50.0, 25.0, 0.0]

#tally up all combinations of possible parameter pairs
combinations = [(i, j) for i in exploring_start_steps, j in r_check_vals]

#split up chunks/jobs
chunk_size = ceil(Int, length(combinations) / 5)  # 5 SLURM array tasks
chunks = [combinations[(i-1)*chunk_size+1:min(i*chunk_size, end)] for i in 1:5]
my_chunk = chunks[job_id]

#params combinations to be ran by chunk
for (expl_step, r_val) in my_chunk
    name = "expl_$(expl_step)_r_$(r_val)_job_$(job_id)"
    SimulationSpace.run_batch(
        rad_sim,
        robot_starts;
        exploring_start=true,
        num_exploring_start_steps=expl_step,
        r_check=r_val,
        filename=name
    )
end

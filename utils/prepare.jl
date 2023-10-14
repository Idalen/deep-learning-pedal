using ArgParse
using WAV
using Statistics
using JLD2
using FileIO


function normalize(data)
    data_max = maximum(data)
    data_min = minimum(data)
    data_norm = max(data_max, abs(data_min))
    return data / data_norm
end

function prepare(in_file, out_file, normalize_flag=true, sample_time=100e-3, data_dir = "./data/")
    in_data, in_rate = wavread(in_file)
    out_data, out_rate = wavread(out_file)

    @assert in_rate == out_rate "in_file and out_file must have the same sample rate"
    

    # Trim the length of audio to equal the smaller WAV file
    if prod(size(in_data)) > prod(size(out_data))
        println("Trimming input audio to match output audio")
        in_data = in_data[1:prod(size(out_data))]
    end
    if prod(size(out_data)) > prod(size(in_data))
        println("Trimming output audio to match input audio")
        out_data = out_data[1:prod(size(in_data))]
    end

    # If stereo data, use channel 1
    if ndims(in_data) > 1
        println("[WARNING] Stereo data detected for in_data, only using first channel (left channel)")
        in_data = in_data[:, 1]
    end
    if ndims(out_data) > 1
        println("[WARNING] Stereo data detected for out_data, only using first channel (left channel)")
        out_data = out_data[:, 1]
    end

    # Convert PCM16 to FP32
    if typeof(in_data[1]) == Int16
        in_data = Float32.(in_data) / 32767
        println("In data converted from PCM16 to FP32")
    end
    if typeof(out_data[1]) == Int16
        out_data = Float32.(out_data) / 32767
        println("Out data converted from PCM16 to FP32")
    end

    # Normalize data
    if normalize_flag == true
        in_data = normalize(in_data)
        out_data = normalize(out_data)
    end

    

    sample_size = Int(in_rate * sample_time)
    length = prod(size(in_data)) - mod(prod(size(in_data)), sample_size)

    x = reshape(in_data[1:length], (sample_size, 1, Int(length / sample_size)))
    y = reshape(out_data[1:length], (sample_size, 1, Int(length / sample_size)))

    split = d -> (d[1:Int(prod(size(d)) * 0.6)], d[Int(prod(size(d)) * 0.6)+1:Int(prod(size(d)) * 0.8)], d[Int(prod(size(d)) * 0.8)+1:end])

    d = Dict()
    d["x_train"], d["x_valid"], d["x_test"] = split(x)
    d["y_train"], d["y_valid"], d["y_test"] = split(y)
    d["mean"], d["std"] = mean(d["x_train"]), std(d["x_train"])

    # Standardize
    for key in ["x_train", "x_valid", "x_test"]
        d[key] = (d[key] .- d["mean"]) ./ d["std"]
    end

    if !isdir(data_dir)
        mkpath(data_dir)
    end

    save(joinpath(data_dir, "data.jld2"), d)
end

function main()
    in_file  = "./data/train_in_fp32.wav"
    out_file = "./data/train_out_fp32.wav"

    prepare(in_file, out_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
using Flux
using Flux: onehotbatch
using Statistics: mean
using JLD2
using CUDA
using BSON: @save

include("../utils/prepare.jl")
include("../utils/loss.jl")
    
struct ResidualBlock
    dilation::Int
    num_channels::Int
    kernel_size::Int
    dilated_conv::Conv
    residual_conv::Conv
    skip_conv::Conv
end

function ResidualBlock(dilation, num_channels, kernel_size)
    dilated_conv = Conv((1,), num_channels => num_channels, dilation=dilation, pad=(kernel_size - 1) * dilation)
    residual_conv = Conv((1,), num_channels => num_channels)
    skip_conv = Conv((1,), num_channels => num_channels)
    
    return ResidualBlock(dilation, num_channels, kernel_size, dilated_conv, residual_conv, skip_conv)
end

function (rb::ResidualBlock)(x)
    

    dilated_output = rb.dilated_conv(x) 
    dilated_output = tanh.(dilated_output) .* relu.(dilated_output)
    #println("DILATED OUTPUT ", size(dilated_output))

    skip_output = rb.skip_conv(dilated_output)
    #println("SKIP ", size(skip_output))
    
    residual_output = rb.residual_conv(dilated_output)    
    #println("RESIDUAL ", size(residual_output))
    output = x + residual_output
    
    return output, skip_output
end

struct WaveNet
    num_blocks::Int
    num_layers::Int
    num_channels::Int
    num_classes::Int
    entry_conv::Conv
    residual_blocks::Vector{ResidualBlock}
    relu::typeof(relu)
    tanh::typeof(tanh_fast)
    skip_conv::Conv
    output_conv1::Conv
    output_conv2::Conv
end

function WaveNet(num_blocks, num_layers, num_channels, num_classes, kernel_size=2)
    entry_conv = Conv((kernel_size,), num_classes => num_channels )
    residual_blocks = ResidualBlock[]

    for _ in 1:num_blocks
        for layer in 1:num_layers
            dilation = 2^(layer-1)
            push!(residual_blocks, ResidualBlock(dilation, num_channels, kernel_size))
        end
    end

    relu = Flux.relu
    tanh = Flux.tanh_fast
    skip_conv = Conv((1,), num_channels => num_channels)
    output_conv1 = Conv((1,), num_channels => num_channels)
    output_conv2 = Conv((1,), num_channels => num_classes)

    return WaveNet(num_blocks, num_layers, num_channels, num_classes, entry_conv, residual_blocks, relu, tanh, skip_conv, output_conv1, output_conv2)
end

function (wn::WaveNet)(x)

    #println("INPUT ", size(x))
    x = wn.entry_conv(x)
    #println("FIRST CONV ", size(x))
    skip_sum = zeros(Float32, size(x)[1], wn.num_channels, 1)

    for (i, block) in enumerate(wn.residual_blocks)
        x, skip = block(x)
        skip_sum += skip
    end

    x = wn.relu(skip_sum)
    x = wn.skip_conv(x)
    x = wn.relu(x)
    x = wn.output_conv1(x)
    x = wn.relu(x) 
    x = wn.output_conv2(x)

    return x
end

Flux.@functor ResidualBlock
Flux.@functor WaveNet

function train!(model::WaveNet, in_file, out_file, batch_size, lr, epochs)
    

    opt = Flux.setup(Adam(lr), model) 

    #prepare(in_file, out_file)
    data = JLD2.load("./data/data.jld2")

    x_train, y_train, x_valid, y_valid, x_test, y_test =
    data["x_train"], data["y_train"], data["x_valid"], data["y_valid"], data["x_test"], data["y_test"]

    train_num_batches = size(x_train)[1]/batch_size

    train_data = Flux.Data.DataLoader((x_train, y_train), batchsize=batch_size) #|> gpu
    valid_data = Flux.Data.DataLoader((x_valid, y_valid), batchsize=batch_size) #|> gpu

    loss(ŷ, y) = error_to_signal(ŷ, y)

    best_loss = Inf     

    for epoch in 1:epochs

        for (i,data) in enumerate(train_data)
            
            # Unpack this element (for supervised training):
            x, y= data
            x = x[:,:,1:1] 
            y = y[:,:,1:1] 
            
            
            # Calculate the gradient of the objective
            # with respect to the parameters within the model:
            ∇ = Flux.gradient(model) do m
                ŷ = m(x)
                loss(ŷ, y)
            end

            # Update the parameters so as to reduce the objective,
            # according the chosen optimisation rule:
            Flux.update!(opt, model, ∇[1])
            
            #println("PARAMS ", Flux.params(model))
            if i+1 > train_num_batches
                break
            end

        end
        
        train_loss = mean(loss(model(x_train[:,:,1:1]), y_train))
        valid_loss = mean(loss(model(x_valid[:,:,1:1]), y_valid))

        best_model = undef
        if valid_loss < best_loss
            best_loss = valid_loss
            best_model = model #|> cpu
        end


        println("Epoch $epoch | Train Loss: $train_loss, Valid Loss: $valid_loss")
    end

    @save "model.bson" best_model
end

function main()
    in_file  = "./data/train_in_fp32.wav"
    out_file = "./data/train_out_fp32.wav"

    batch_size = 512
    lr = 1e-2
    epochs = 40

    num_blocks = 16
    num_layers = 8
    num_channels = 2
    num_classes = 1
    kernel_size  = 1

    if CUDA.functional() 
        println("GPU available. Model will be trained in GPU")
    else
        println("GPU not available. Model will be trained in CPU")
    end

    model = WaveNet(num_blocks, num_layers, num_channels, num_classes, kernel_size) #|> gpu

    train!(model,in_file, out_file, batch_size, lr, epochs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
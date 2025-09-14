import torch
import torch.nn as nn
import torch.nn.functional as F


# Network architecture constants
CONV_KERNEL_1X1 = 1
CONV_KERNEL_3X3 = (1, 3, 3)
CONV_KERNEL_5X5 = (1, 5, 5)
CONV_KERNEL_7X7 = (1, 7, 7)

PADDING_NONE = 0
PADDING_1 = (0, 1, 1)
PADDING_2 = (0, 2, 2)
PADDING_3 = (0, 3, 3)

STRIDE_1 = 1
STRIDE_2x2 = (1, 2, 2)

POOL_KERNEL_3X3 = (1, 3, 3)
INCEPTION_BRANCHES = 4


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        
        # Branch 1: 1x1x1 conv
        self.branch1 = self._create_conv_branch(
            in_channels, out_channels, CONV_KERNEL_1X1, PADDING_NONE
        )
        
        # Branch 2: 1x1x1 conv -> 1x3x3 conv
        self.branch2 = self._create_sequential_conv_branch(
            in_channels, out_channels, CONV_KERNEL_3X3, PADDING_1
        )
        
        # Branch 3: 1x1x1 conv -> 1x5x5 conv
        self.branch3 = self._create_sequential_conv_branch(
            in_channels, out_channels, CONV_KERNEL_5X5, PADDING_2
        )
        
        # Branch 4: MaxPool -> 1x1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(
                kernel_size=POOL_KERNEL_3X3,
                stride=STRIDE_1, padding=PADDING_1
            ),
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=CONV_KERNEL_1X1,
                padding=PADDING_NONE, stride=STRIDE_1, bias=True
            ),
            nn.ReLU()
        )
    
    def _create_conv_branch(self, in_channels, out_channels, kernel_size, padding):
        """
        Create a simple convolution branch with ReLU activation.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            
        Returns:
            Sequential module containing conv + ReLU
        """
        return nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=padding, stride=STRIDE_1, bias=True
            ),
            nn.ReLU()
        )
    
    def _create_sequential_conv_branch(self, in_channels, out_channels, second_kernel, second_padding):
        """
        Create a sequential convolution branch: 1x1 conv -> larger conv + ReLU.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            second_kernel: Kernel size for the second convolution
            second_padding: Padding for the second convolution
            
        Returns:
            Sequential module containing 1x1 conv + ReLU + larger conv + ReLU
        """
        return nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=CONV_KERNEL_1X1,
                padding=PADDING_NONE, stride=STRIDE_1, bias=True
            ),
            nn.ReLU(),
            nn.Conv3d(
                out_channels, out_channels,
                kernel_size=second_kernel,
                padding=second_padding, stride=STRIDE_1, bias=True
            ),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Execute all branches and concatenate
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


class InceptionModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModel, self).__init__()
        # Initial conv layer
        self.initial_layers = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=CONV_KERNEL_7X7,
                padding=PADDING_3, stride=STRIDE_1, bias=True
            ),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=POOL_KERNEL_3X3,
                padding=PADDING_1, stride=STRIDE_2x2
            )
        )
        
        # Three inception blocks with pooling
        self.inception1 = InceptionBlock(out_channels, out_channels)
        self.pool1 = nn.MaxPool3d(
            kernel_size=POOL_KERNEL_3X3,
            stride=STRIDE_2x2, padding=PADDING_1
        )

        self.inception2 = InceptionBlock(out_channels * INCEPTION_BRANCHES, out_channels)
        self.pool2 = nn.MaxPool3d(
            kernel_size=POOL_KERNEL_3X3,
            padding=PADDING_1, stride=STRIDE_2x2
        )
        
        self.inception3 = InceptionBlock(out_channels * INCEPTION_BRANCHES, out_channels)
        self.pool3 = nn.MaxPool3d(
            kernel_size=POOL_KERNEL_3X3,
            padding=PADDING_1, stride=STRIDE_2x2
        )

    def forward(self, x):
        # Chain operations without intermediate variables
        x = self.initial_layers(x)
        x = self.pool1(self.inception1(x))
        x = self.pool2(self.inception2(x))
        return self.pool3(self.inception3(x))


class InceptionLSTMModel(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, hidden_size):
        super(InceptionLSTMModel, self).__init__()

        # Validate input parameters
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(f"Channels must be positive, got in_channels={in_channels}, out_channels={out_channels}")
        if image_size <= 0 or not isinstance(image_size, int):
            raise ValueError(f"Image size must be a positive integer, got {image_size}")
        if hidden_size <= 0:
            raise ValueError(f"Hidden size must be positive, got {hidden_size}")

        self.inception = InceptionModel(in_channels, out_channels)
        
        # Calculate feature size after convolutions
        # InceptionModel has 4 pooling operations with stride 2: initial_pool + pool1 + pool2 + pool3
        num_pooling_operations = 4
        pooling_stride = 2
        final_spatial_size = image_size // (pooling_stride ** num_pooling_operations)
        
        # Check if image size is valid for pooling operations
        if final_spatial_size <= 0:
            raise ValueError(f"Image size {image_size} is too small for {num_pooling_operations} pooling operations with stride {pooling_stride}")
        
        final_channels = out_channels * INCEPTION_BRANCHES
        lstm_input_size = final_channels * final_spatial_size * final_spatial_size
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                           hidden_size=hidden_size, batch_first=True)
        self.act = nn.ReLU()

    def forward(self, x):
        # Validate input tensor dimensions
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input tensor (batch, channels, seq_len, height, width), got {x.dim()}D tensor")
        
        batch_size, channels, seq_len, height, width = x.size()
        
        if seq_len <= 0:
            raise ValueError(f"Sequence length must be positive, got {seq_len}")
        if height != width:
            raise ValueError(f"Expected square images, got height={height}, width={width}")

        # Process through inception and reshape for LSTM
        inception_out = self.inception(x)
        batch_size, channels, seq_len, height, width = inception_out.size()

        # Use reshape instead of view for better memory efficiency
        lstm_input = inception_out.permute(0, 2, 1, 3, 4).reshape(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(lstm_input)
        return self.act(lstm_out[:, -1, :])


class LinearBlock(nn.Module):
    def __init__(self, input_variables, output_variables):
        super(LinearBlock, self).__init__()
        
        # Validate input parameters
        if input_variables <= 0 or output_variables <= 0:
            raise ValueError(f"Input and output variables must be positive, got input={input_variables}, output={output_variables}")
        
        self.layers = nn.Sequential(
            nn.Linear(input_variables, output_variables),
            nn.ReLU(),
            nn.Linear(output_variables, output_variables),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)
    

class LinearModel(nn.Module):
    def __init__(self, num_input_variables, input_sequence_length, num_linear_output):
        super(LinearModel, self).__init__()
        
        # Validate input parameters
        if num_input_variables <= 0:
            raise ValueError(f"Number of input variables must be positive, got {num_input_variables}")
        if input_sequence_length <= 0:
            raise ValueError(f"Input sequence length must be positive, got {input_sequence_length}")
        if num_linear_output <= 0:
            raise ValueError(f"Number of linear output units must be positive, got {num_linear_output}")
        
        self.num_input_variables = num_input_variables
        self.input_sequence_length = input_sequence_length

        # Create blocks using ModuleList for better organization
        self.linear_blocks = nn.ModuleList([
            LinearBlock(input_sequence_length, num_linear_output)
            for _ in range(num_input_variables)
        ])

    def forward(self, x):
        # Validate input tensor dimensions
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (batch, sequence, variables), got {x.dim()}D tensor")
        
        batch_size, seq_len, num_vars = x.size()
        
        if seq_len != self.input_sequence_length:
            raise ValueError(f"Expected sequence length {self.input_sequence_length}, got {seq_len}")
        if num_vars != self.num_input_variables:
            raise ValueError(f"Expected {self.num_input_variables} variables, got {num_vars}")

        # Process each variable and concatenate directly
        return torch.cat([
            self.linear_blocks[variable_idx](x[:, :, variable_idx])
            for variable_idx in range(self.num_input_variables)
        ], dim=-1)


class MultiModalModel(nn.Module):
    def __init__(
        self, num_input_variables, input_sequence_length,
        num_target_variables, target_sequence_length,
        num_linear_output,
        inception_in_channels, inception_out_channels,
        inception_in_image_size, inception_in_image_frames,
        lstm_hidden_size
        ) :
        super(MultiModalModel, self).__init__()

        # Validate input parameters
        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        self.linear_model = LinearModel(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            num_linear_output=num_linear_output)

        self.inception_model = InceptionLSTMModel(
            in_channels=inception_in_channels,
            out_channels=inception_out_channels,
            image_size=inception_in_image_size,
            hidden_size=lstm_hidden_size)

        self.combine_layer = nn.Linear(
            lstm_hidden_size + num_linear_output * num_input_variables, num_linear_output)
        self.output_layer = nn.Linear(num_linear_output, num_target_variables * target_sequence_length)

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

    def forward(self, solar_wind_input, image_input):
        """
        Forward pass through the multi-modal model.
        
        Args:
            solar_wind_input (torch.Tensor): Solar wind data of shape (batch, sequence, variables).
            image_input (torch.Tensor): Image data of shape (batch, channels, seq_len, height, width).
            
        Returns:
            torch.Tensor: Predicted output of shape (batch, target_sequence_length, num_target_variables).
            
        Raises:
            ValueError: If inputs are None or batch sizes don't match.
        """
        # Validate input tensors
        if solar_wind_input is None or image_input is None:
            raise ValueError("Both solar_wind_input and image_input must be provided")
        
        if solar_wind_input.size(0) != image_input.size(0):
            raise ValueError(f"Batch sizes must match: solar_wind={solar_wind_input.size(0)}, image={image_input.size(0)}")

        # Process inputs and combine features
        combined_features = torch.cat([
            self.inception_model(image_input),
            self.linear_model(solar_wind_input)
        ], dim=1)
        
        # Apply combine layer and output layer, then reshape
        output = self.output_layer(F.relu(self.combine_layer(combined_features)))
        return output.reshape(output.size(0), self.target_sequence_length, self.num_target_variables)


# Usage example:
def create_model(options):
    """
    Create MultiModalModel instance from configuration options.
    
    Args:
        options: Configuration object containing model parameters
        
    Returns:
        MultiModalModel instance
    """
    return MultiModalModel(
        num_input_variables=options.num_input_variables,
        input_sequence_length=options.input_sequence_length,
        num_target_variables=options.num_target_variables,
        target_sequence_length=options.target_sequence_length,
        num_linear_output=options.num_linear_output,
        inception_in_channels=options.inception_in_channels,
        inception_out_channels=options.inception_out_channels,
        inception_in_image_size=options.inception_in_image_size,
        inception_in_image_frames=options.inception_in_image_frames,
        lstm_hidden_size=options.lstm_hidden_size
    )
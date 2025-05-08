def parse_sequence_length(sequence_length):
    values = tuple(map(int, sequence_length.split(",")))
    if len(values) != 2:
        raise ValueError(
            "Sequence length must be in the format 'int,int' (e.g., 5,10)."
        )
    if values[0] > values[1]:
        raise ValueError(
            "The first element of sequence_length must be less than or equal to the second."
        )
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError(
            "Both elements of sequence_length must be positive integers."
        )
    return values[0], values[1]
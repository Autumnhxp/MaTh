def write_with_path(writer, rep_output, frame_padding=4, image_output_format="png"):
    """
    A helper function to wrap the BasicWriter's `write` method and infer the saved file paths for rgb, normals,
    and distance_to_image_plane.

    Args:
        writer: An instance of BasicWriter.
        rep_output: The replicator output to be written.
        frame_padding: The number of digits for frame padding in filenames.
        image_output_format: The image output format (e.g., "png").

    Returns:
        A dictionary containing paths for rgb, normals, and distance_to_image_plane files.
    """
    # Call the original write method
    writer.write(rep_output)

    # Extract sequence ID and frame ID from writer's state
    sequence_id = writer._sequence_id  # Assuming the private attribute is accessible
    frame_id = writer._frame_id - 1   # `_frame_id` is incremented after write

    # Generate paths based on annotators
    output_paths = {
        "rgb": None,
        "normals": None,
        "distance_to_image_plane": None
    }
    output_dir = writer._output_dir  # Root output directory

    for annotator_name in rep_output["annotators"].keys():
        if annotator_name == "rgb":
            file_path = f"{output_dir}/rgb_{sequence_id}{frame_id:0{frame_padding}}.{image_output_format}"
            output_paths["rgb"] = file_path
        elif annotator_name == "normals":
            file_path = f"{output_dir}/normals_{sequence_id}{frame_id:0{frame_padding}}.{image_output_format}"
            output_paths["normals"] = file_path
        elif annotator_name == "distance_to_image_plane":
            file_path = f"{output_dir}/distance_to_image_plane_{sequence_id}{frame_id:0{frame_padding}}.npy"
            output_paths["distance_to_image_plane"] = file_path

    return output_paths

def get_line_clusters(bites):
    return {line_id: bite_id for bite_id, bite in enumerate(bites) for line_id in bite}

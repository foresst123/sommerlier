def detect_overlapping_segments(segment_list: list, overlap_threshold: float = 0.2, logger=None) -> list:
    """Detect segments that overlap for more than overlap_threshold seconds."""
    overlapping_pairs = []
    sorted_segments = sorted(segment_list, key=lambda x: x['start'])

    for i in range(len(sorted_segments)):
        for j in range(i + 1, len(sorted_segments)):
            seg1 = sorted_segments[i]
            seg2 = sorted_segments[j]

            if seg2['start'] >= seg1['end']:
                break

            overlap_start = max(seg1['start'], seg2['start'])
            overlap_end = min(seg1['end'], seg2['end'])
            overlap_duration = overlap_end - overlap_start

            if overlap_duration >= overlap_threshold:
                overlapping_pairs.append({
                    'seg1': seg1,
                    'seg2': seg2,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_duration': overlap_duration
                })
                if logger:
                    logger.info(f"Overlap detected: {overlap_duration:.2f}s between "
                               f"[{seg1['start']:.2f}-{seg1['end']:.2f}] and "
                               f"[{seg2['start']:.2f}-{seg2['end']:.2f}]")

    return overlapping_pairs

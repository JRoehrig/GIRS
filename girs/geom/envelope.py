

def is_intersect_envelope(env0, env1):
    return not (env0[0] > env1[1] or env0[1] < env1[0] or env0[2] > env1[3] or env0[3] < env1[2])


def is_intersect_envelopes_any(env, *envs):
    for env1 in envs:
        if is_intersect_envelope(env, env1):
            return True
    return False


def is_intersect_envelopes_all(env, *envs):
    return all([is_intersect_envelope(env, env1) for env1 in envs])


def scale_envelope(envelope, scale=0.0):
    xmin, xmax, ymin, ymax = envelope
    if scale and scale != 0.0:
        dx, dy = xmax - xmin, ymax - ymin
        xmin -= dx * scale
        xmax += dx * scale
        ymin -= dy * scale
        ymax += dy * scale
    return xmin, xmax, ymin, ymax


def buffer_envelope(envelope, distance):
    try:
        return envelope[0] - distance, envelope[1] + distance, \
               envelope[2] - distance, envelope[3] + distance
    except TypeError:
        return envelope[0] - distance[0], envelope[1] + distance[0], \
               envelope[2] - distance[1], envelope[3] + distance[1]


def merge_envelopes(envelopes, extent_type='intersection'):
    """Return the extent of a list of envelopes.

    Return the extent of the union or intersection of a list of envelopes

    :param envelopes: list of envelopes or raster filenames (also mixed)
    :type envelopes: list, tuple, iterator
    :param extent_type: intersection or union
    :return: (xmin, xmax, ymin, ymax) in world coordinates
    """
    et = extent_type.lower()
    xmin, xmax, ymin, ymax = envelopes[0]
    if len(envelopes) > 1:
        if et == 'intersection':
            for xmin0, xmax0, ymin0, ymax0 in envelopes[1:]:
                xmin = max(xmin, xmin0)
                xmax = min(xmax, xmax0)
                ymin = max(ymin, ymin0)
                ymax = min(ymax, ymax0)
        elif et == 'union':
            for xmin0, xmax0, ymin0, ymax0 in envelopes[1:]:
                xmin = min(xmin, xmin0)
                xmax = max(xmax, xmax0)
                ymin = min(ymin, ymin0)
                ymax = max(ymax, ymax0)
        else:
            msg = 'extent_type {} unknown'.format(extent_type)
            raise TypeError(msg)
    return xmin, xmax, ymin, ymax



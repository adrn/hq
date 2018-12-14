__all__ = ['paged_query']

def paged_query(query, page_size=1000):
    """
    """
    n_tot = query.count()

    n_pages = n_tot // page_size
    if n_tot % page_size:
        n_pages += 1

    for page in range(n_pages):
        q = query.limit(page_size)

        if page:
            q = q.offset(page*page_size)

        yield q

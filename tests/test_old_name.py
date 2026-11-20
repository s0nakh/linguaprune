import warnings


def test_old_name_still_imports_the_same_modules():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import linguaprune
        import linguaprune.quant as old_quant
    import linguasqueeze
    import linguasqueeze.quant as new_quant

    assert old_quant is new_quant
    assert linguaprune.__version__ == linguasqueeze.__version__
    assert linguaprune.allocate_like is linguasqueeze.allocate_like

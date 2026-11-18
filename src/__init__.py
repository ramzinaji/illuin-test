import sys

# Le code utilise la syntaxe `str | None` (Python 3.10+) ; sous 3.9, l'erreur serait obscure
if sys.version_info < (3, 10):  # noqa: UP036 (contrôle voulu : exécuté avant le reste du code)
    raise RuntimeError(
        f"Python 3.10 ou plus récent est requis (version utilisée : {sys.version.split()[0]})."
    )

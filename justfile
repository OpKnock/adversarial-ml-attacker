default:
    just test

test:
    python -m pytest

install:
    python -m pip install -e .

demo:
    python -m advml campaign --out reports

clean:
    -rmdir /s /q reports 2>nul

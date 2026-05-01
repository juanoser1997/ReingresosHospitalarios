# Guia de Contribucion

## Entorno de Desarrollo

```bash
git clone https://github.com/<tu-usuario>/mlops-alto-costo.git
cd mlops-alto-costo

# Instalar uv si no lo tienes
pip install uv

# Crear entorno e instalar todas las dependencias (incluyendo dev)
uv sync --all-groups

# Activar el entorno
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Instalar los hooks de pre-commit
uv run pre-commit install
```

## Flujo de Trabajo con Git

1. Crear una rama desde `main`:

```bash
git checkout -b feat/nombre-de-la-funcionalidad
```

2. Hacer cambios y commitear siguiendo Conventional Commits:

```
feat(data): agregar validacion de schema de entrada
fix(api): corregir manejo de nulos en endpoint /predict
docs(readme): actualizar instrucciones de despliegue
test(models): agregar tests para la transformacion del target
```

3. Asegurarse de que los tests pasen:

```bash
uv run pytest tests/unit/ -v
```

4. Abrir un Pull Request hacia `main`.

## Convencion de Mensajes de Commit

```
<tipo>(<scope>): <descripcion corta en imperativo>
```

Tipos: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`

## Estandares de Codigo

El proyecto usa **ruff** para linting y formateo. Se ejecuta automaticamente
en cada commit via pre-commit. Para ejecutarlo manualmente:

```bash
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
```

## Estructura de Tests

- `tests/unit/`: tests que no requieren datos en disco ni servicios externos.
- `tests/integration/`: tests que requieren que el pipeline de datos haya corrido
  y los archivos procesados existan en `data/processed/`.

Los tests de integracion se marcan con `@pytest.mark.integration` y se excluyen
del CI rapido. Se ejecutan antes de cada release.

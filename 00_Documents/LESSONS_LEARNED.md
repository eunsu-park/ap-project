# Lessons Learned

Record of mistakes, insights, and best practices discovered during the project.

## Template

```
### [YYYY-MM-DD] Title
- **Category**: (e.g., Data, Model, Pipeline, Environment, etc.)
- **Problem**: What went wrong or what was discovered
- **Solution**: How it was resolved
- **Takeaway**: What to do differently next time
```

## Entries

### [2026-02-14] aiapy 0.11.0 API breaking change
- **Category**: Environment
- **Problem**: `from aiapy.calibrate.util import get_correction_table` raises `ModuleNotFoundError` in aiapy 0.11.0
- **Solution**: Renamed `util` → `utils` (`from aiapy.calibrate.utils import get_correction_table`)
- **Takeaway**: Pin dependency versions or check changelogs when upgrading packages

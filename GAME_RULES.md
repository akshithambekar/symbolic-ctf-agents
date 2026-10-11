# MCTF 2026 Rules

Source: [official rules](https://www.mctf26.com/rules), reviewed 2026-10-11; source governs.

## Game Setup

Red and Blue each control three autonomous boats. Matches last 10 minutes.

- Field: 160 × 80 m, divided into two 80 × 80 m territories. Boats can enter either half.
- Each team has two quarter-circle home bases at its corners furthest from midfield.
- Capture/untag zones are described as “10 m diameter” from corners; the precise geometry needs clarification.

## Tagging and Recovery

A boat tags an opponent within 10 m when both occupy the tagging team's territory.

Tagged boats cannot tag or interact with flags. A carried flag immediately returns to its owner's base. The controller drives tagged boats home, then restores agent control upon untagging.

Leaving the field causes an automatic tag and immediate return to the nearest home base for recovery.

## Flag Events

- **Grab:** An untagged boat approaches within 10 m of the opponent's available flag.
- **Capture:** The untagged carrier returns to either home base with that flag.

## Powerplay

Each team receives one two-minute Powerplay; opposing periods cannot overlap. The controller disables one opponent's movement, tagging, and flag interaction, then automatically re-enables it afterward.

## Scoring

```text
Team score = team captures − opponent captures
```

Ties favor more grabs, then fewer collisions. A collision means coming within 4 m of any teammate or opponent.

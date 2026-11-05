# Phase 0 notes

Phase 0 complete on 16 Oct 2026. These numbers set the targets for Phase 1.

## Laptop

- CPU: AMD Ryzen 5 7530U, 6 cores / 12 threads
- RAM: 15.4 GB
- Disk: NVMe SSD, about 162 GB free
- Windows 11 Home

## Setup check (`ks check`)

- All lines OK: Python 3.11.9, ExifTool 13.59, LibRaw 0.22.1, all libraries
- JPEG decode, 24 MP: full 190 ms, reduced 120 ms (1.6x faster)
- Tests: 11 passing (scanner + matching)

## First real wedding: Neel & Tara (Bali)

Client drive: 2 TB external hard disk, read in place (never copied, never written to).

- Files: 12,926 (11,853 originals in "Raw image's" + 1,073 copies in "Selected image's")
- Types: 12,889 JPEG, 37 DNG (drone). The originals are camera JPEGs, not RAW files
- Cameras: Nikon Z9 (8,847), Nikon Z8 (2,004), Canon EOS R6 Mark III (948),
  Nikon Z6III (819), Leica Q (208), Canon EOS R5 (63), DJI drone FC8282 (37)
- Damaged files: 0. Missing capture time: 0
- Metadata speed, `ks list` on the drive: 14 photos/s (15.4 min). Baseline for Phase 1 to beat
  (ExifTool `-fast2` mode added afterwards)

## Labels (`ks match`)

- Matched 1,071 of 1,073 selected photos to their originals in 1.7 s
  - by file name: 605
  - by file name + size: 466 (several cameras reuse the same file names)
- Not matched: 2, both typo names (`DSC_0342.JPGG.JPG`, `DSC_0509.JPGG.JPG`)
- Selected copies are exact copies of the originals (same size), not edited versions
- Labels saved on this laptop: `data\labels\neel_tara_bali.csv`

| Event | Photos | Selected | Kept |
| --- | --- | --- | --- |
| 27.08.2026 Welcome & Bhajan | 8,173 | 515 | 6.3% |
| 29.08.2026 Dinner | 3,680 | 556 | 15.1% |
| Total | 11,853 | 1,071 | 9.0% |

## Findings that shape the next phases

1. The studio is far stricter at some events than others, so every judgement must be relative to its own event.
2. 194 photos are dated 31 Dec 2014 / 1 Jan 2015: one camera's clock was never set. Clock alignment (Phase 3) is required.
3. 441 photos are dated 26 Aug, probably a camera on India time while shooting in Bali.
4. Honest test plan: train on Welcome & Bhajan, test on Dinner.
5. Several photographers' cameras reuse file names, so a file name alone never identifies a photo.
6. Two selected files had typo names; the matcher should handle `.JPGG.JPG`-style names (next update).

## Still to send

- [ ] Client: Mac models, camera list (now partly known), how they cull today, permission note
- [ ] Vijay Sir: hours per week, a Mac for the week-2 packaging test

//! Help screen layout.
//!
//! This module owns no content of its own. It renders what is already declared
//! elsewhere, so the help screen cannot fall behind the program:
//!
//! * keyboard and mouse bindings come from `keymap::KEYMAP`
//! * command line subcommands come from the `clap` definition in `cli.rs`
//! * shell functions come from the `shell/psychic.zsh` we embed and ship
//!
//! It exposes pure layout functions ([`blocks`], [`lay_out`]) that turn that
//! content into [`HelpLine`]s. `render.rs` styles and draws them; the tests here
//! check the layout without a terminal.

use crate::keymap::{self, Section};

/// One row of the help screen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The keys, or the command to type.
    pub keys: String,
    pub description: String,
}

/// A titled group of rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub title: String,
    pub entries: Vec<Entry>,
}

impl Block {
    /// Lines this block occupies, excluding the blank line separating blocks.
    fn height(&self) -> usize {
        1 + self.entries.len()
    }
}

/// A laid-out line, ready to be styled and drawn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HelpLine {
    /// A block heading.
    Title(String),
    /// A row. `keys` is padded to the shared key column width.
    Entry { keys: String, description: String },
    /// Spacing between blocks.
    Blank,
}

/// Gap between the key column and the description column.
const KEY_GAP: usize = 2;

/// Everything the help screen shows, in display order.
pub fn blocks() -> Vec<Block> {
    let mut blocks: Vec<Block> = Section::ALL
        .iter()
        .map(|section| Block {
            title: section.title().to_string(),
            entries: keymap::bindings_in(*section)
                .map(|binding| Entry {
                    keys: binding.keys(),
                    description: binding.description().to_string(),
                })
                .collect(),
        })
        .collect();

    blocks.push(Block {
        title: "Shell commands".to_string(),
        entries: shell_commands(SHELL_INTEGRATION),
    });
    blocks.push(Block {
        title: "Command line (psychic ...)".to_string(),
        entries: cli_commands(),
    });

    blocks
}

/// The shell integration script we embed, ship via `psychic zsh`, and read here.
const SHELL_INTEGRATION: &str = include_str!("../shell/psychic.zsh");

/// The functions `eval "$(psychic zsh)"` defines, read out of the script itself.
///
/// A function's description is the first line of the comment block above it, so
/// the help screen stays right as long as the script stays commented.
fn shell_commands(script: &str) -> Vec<Entry> {
    let lines: Vec<&str> = script.lines().collect();
    let mut entries = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let Some(name) = line.strip_suffix("() {") else {
            continue;
        };
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric()) {
            continue;
        }

        // Walk back over the comment block directly above the function.
        let mut description = String::new();
        let mut j = i;
        while j > 0 {
            let above = lines[j - 1].trim();
            if let Some(comment) = above.strip_prefix("# ") {
                description = comment.to_string();
                j -= 1;
            } else {
                break;
            }
        }

        entries.push(Entry {
            keys: name.to_string(),
            description: description.to_lowercase(),
        });
    }

    entries
}

/// Subcommands, read from the clap definition in `cli.rs`.
fn cli_commands() -> Vec<Entry> {
    use clap::CommandFactory;

    crate::cli::Cli::command()
        .get_subcommands()
        .map(|command| Entry {
            keys: command.get_name().to_string(),
            description: command
                .get_about()
                .map(|about| about.to_string().to_lowercase())
                .unwrap_or_default(),
        })
        .collect()
}

/// Width of the key column: the widest key string anywhere on the screen.
///
/// Shared by every column so the two halves of the screen line up with each
/// other, not just within themselves.
fn key_column_width(blocks: &[Block]) -> usize {
    blocks
        .iter()
        .flat_map(|block| block.entries.iter())
        .map(|entry| entry.keys.len())
        .max()
        .unwrap_or(0)
}

/// Width a column needs to show every description in full.
fn natural_column_width(blocks: &[Block]) -> usize {
    let description_width = blocks
        .iter()
        .flat_map(|block| block.entries.iter())
        .map(|entry| entry.description.len())
        .max()
        .unwrap_or(0);

    key_column_width(blocks) + KEY_GAP + description_width
}

/// Space between the two columns.
const GUTTER: usize = 3;

/// A description shorter than this is not worth a second column.
const MIN_DESCRIPTION: usize = 22;

/// A help screen laid out for a given width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelpLayout {
    /// One entry per column, left to right.
    pub columns: Vec<Vec<HelpLine>>,
    /// Width of every column. Columns are equal so they can be aligned simply.
    pub column_width: usize,
    /// Space between columns.
    pub gutter: usize,
}

impl HelpLayout {
    /// Total width the layout occupies.
    pub fn width(&self) -> usize {
        match self.columns.len() {
            0 => 0,
            n => self.column_width * n + self.gutter * (n - 1),
        }
    }

    /// Lines in the tallest column: how much vertical room the layout wants.
    pub fn height(&self) -> usize {
        self.columns
            .iter()
            .map(|column| column.len())
            .max()
            .unwrap_or(0)
    }
}

/// Lay the help screen out to fit `available_width`.
///
/// Uses two columns when they would each still be readable, and never asks for
/// more width than it can use. Descriptions are truncated to the column width,
/// so the caller never has to deal with lines that overflow.
pub fn lay_out(blocks: &[Block], available_width: usize) -> HelpLayout {
    assert!(!blocks.is_empty(), "Cannot lay out an empty help screen");

    let key_width = key_column_width(blocks);
    let natural = natural_column_width(blocks);
    let minimum = key_width + KEY_GAP + MIN_DESCRIPTION;

    let two_columns = blocks.len() >= 2 && available_width >= minimum * 2 + GUTTER;

    let (splits, column_width) = if two_columns {
        let split = balanced_split(blocks);
        let width = ((available_width - GUTTER) / 2).min(natural);
        (vec![&blocks[..split], &blocks[split..]], width)
    } else {
        (vec![blocks], available_width.min(natural))
    };

    HelpLayout {
        columns: splits
            .into_iter()
            .map(|blocks| lines_for(blocks, key_width, column_width))
            .collect(),
        column_width,
        gutter: GUTTER,
    }
}

/// Turn blocks into lines, padding keys to `key_width` and fitting `column_width`.
fn lines_for(blocks: &[Block], key_width: usize, column_width: usize) -> Vec<HelpLine> {
    let description_width = column_width.saturating_sub(key_width + KEY_GAP);
    let mut lines = Vec::new();

    for (i, block) in blocks.iter().enumerate() {
        if i > 0 {
            lines.push(HelpLine::Blank);
        }
        lines.push(HelpLine::Title(truncate(&block.title, column_width)));
        for entry in &block.entries {
            lines.push(HelpLine::Entry {
                keys: format!("{:<width$}", entry.keys, width = key_width),
                description: truncate(&entry.description, description_width),
            });
        }
    }

    lines
}

/// Shorten `text` to `width` characters, on a word boundary where one is close.
///
/// Descriptions come from doc comments and shell comments, which can be longer
/// than a column; cutting one is better than sizing the whole screen for it.
fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    if width == 0 {
        return String::new();
    }

    let head: String = text.chars().take(width - 1).collect();

    // Back up to a word boundary, unless that throws away most of the column.
    let end = match head.rfind(' ') {
        Some(space) if space * 5 >= head.len() * 3 => space,
        _ => head.len(),
    };

    format!("{}\u{2026}", &head[..end])
}

/// Index of the first block that belongs in the right column.
///
/// Picks the split whose halves are closest in height, so neither column runs
/// much longer than the other. Blocks keep their order.
fn balanced_split(blocks: &[Block]) -> usize {
    assert!(
        blocks.len() >= 2,
        "A balanced split needs at least two blocks"
    );

    let total: usize = blocks.iter().map(Block::height).sum();

    let mut best_split = 1;
    let mut best_imbalance = usize::MAX;
    let mut left_height = 0;

    // Every candidate split leaves at least one block in each column.
    for (index, block) in blocks.iter().enumerate().take(blocks.len() - 1) {
        left_height += block.height();
        let imbalance = left_height.abs_diff(total - left_height);
        if imbalance < best_imbalance {
            best_imbalance = imbalance;
            best_split = index + 1;
        }
    }

    best_split
}

/// Render lines as plain strings, so layout can be checked without a terminal.
/// The TUI styles the lines instead of going through here.
#[cfg(test)]
pub fn plain_text(lines: &[HelpLine]) -> Vec<String> {
    lines
        .iter()
        .map(|line| match line {
            HelpLine::Title(title) => title.clone(),
            HelpLine::Entry { keys, description } => {
                format!("{}{}{}", keys, " ".repeat(KEY_GAP), description)
            }
            HelpLine::Blank => String::new(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small fixture, so layout tests do not churn when a keybinding changes.
    fn fixture() -> Vec<Block> {
        vec![
            Block {
                title: "One".to_string(),
                entries: vec![
                    Entry {
                        keys: "a".to_string(),
                        description: "first".to_string(),
                    },
                    Entry {
                        keys: "Ctrl-B".to_string(),
                        description: "second".to_string(),
                    },
                ],
            },
            Block {
                title: "Two".to_string(),
                entries: vec![Entry {
                    keys: "c".to_string(),
                    description: "third".to_string(),
                }],
            },
        ]
    }

    /// Widths of the fixture: keys pad to 6 ("Ctrl-B") and descriptions need 6,
    /// so a column wants 14 and a second column is only worth it past 63.
    const FIXTURE_COLUMN_WIDTH: usize = 14;

    #[test]
    fn test_narrow_screen_gets_one_column() {
        let layout = lay_out(&fixture(), 40);
        assert_eq!(layout.columns.len(), 1);
        assert_eq!(layout.column_width, FIXTURE_COLUMN_WIDTH);
        assert_eq!(
            plain_text(&layout.columns[0]),
            vec![
                "One".to_string(),
                "a       first".to_string(),
                "Ctrl-B  second".to_string(),
                "".to_string(),
                "Two".to_string(),
                "c       third".to_string(),
            ],
            "Keys pad to the widest key; blocks are separated by a blank line"
        );
    }

    #[test]
    fn test_wide_screen_gets_two_columns() {
        let layout = lay_out(&fixture(), 100);
        assert_eq!(layout.columns.len(), 2);
        assert_eq!(
            plain_text(&layout.columns[0]),
            vec![
                "One".to_string(),
                "a       first".to_string(),
                "Ctrl-B  second".to_string(),
            ]
        );
        assert_eq!(
            plain_text(&layout.columns[1]),
            vec!["Two".to_string(), "c       third".to_string()],
            "The right column pads keys to the same width as the left"
        );
    }

    #[test]
    fn test_layout_never_asks_for_more_room_than_it_has() {
        // Every width from cramped to roomy, including both sides of the
        // threshold where the second column appears.
        for available in 10..200 {
            let layout = lay_out(&fixture(), available);
            assert!(
                layout.width() <= available.max(FIXTURE_COLUMN_WIDTH),
                "Layout wanted {} columns of {} in {} available",
                layout.columns.len(),
                layout.column_width,
                available
            );
            for line in layout.columns.iter().flatten() {
                let rendered = plain_text(std::slice::from_ref(line));
                assert!(
                    rendered[0].chars().count() <= layout.column_width,
                    "Line {:?} overflows a column of {}",
                    rendered[0],
                    layout.column_width
                );
            }
        }
    }

    #[test]
    fn test_height_is_the_tallest_column() {
        assert_eq!(lay_out(&fixture(), 100).height(), 3);
        assert_eq!(lay_out(&fixture(), 40).height(), 6);
    }

    #[test]
    fn test_truncate_keeps_short_text() {
        assert_eq!(truncate("clear the query", 20), "clear the query");
        assert_eq!(truncate("clear the query", 15), "clear the query");
    }

    #[test]
    fn test_truncate_prefers_a_word_boundary() {
        assert_eq!(
            truncate("generate features for training from events", 30),
            "generate features for\u{2026}",
            "Cutting at a word reads better than cutting mid-word"
        );
    }

    #[test]
    fn test_truncate_cuts_mid_word_rather_than_lose_the_line() {
        assert_eq!(
            truncate("supercalifragilistic expialidocious", 12),
            "supercalifr\u{2026}",
            "Backing up to the space would throw away most of the column"
        );
    }

    #[test]
    fn test_truncate_degenerate_widths() {
        assert_eq!(truncate("anything", 0), "");
        assert_eq!(truncate("anything", 1), "\u{2026}");
    }

    #[test]
    fn test_the_real_screen_fits_a_small_terminal() {
        // 80x24 is the classic floor. The screen may need scrolling there, but
        // it must not be laid out wider than the terminal.
        let layout = lay_out(&blocks(), 76);
        assert!(
            layout.width() <= 76,
            "Help screen wants {} columns of {}",
            layout.columns.len(),
            layout.column_width
        );
    }

    #[test]
    fn test_shell_commands_are_read_from_the_script() {
        let script = "\
# Jump to a directory
# More detail nobody needs
p() {
    :
}

# Directories only
pd() {
    :
}
";
        assert_eq!(
            shell_commands(script),
            vec![
                Entry {
                    keys: "p".to_string(),
                    description: "jump to a directory".to_string(),
                },
                Entry {
                    keys: "pd".to_string(),
                    description: "directories only".to_string(),
                },
            ],
            "The first comment line above a function is its description"
        );
    }

    #[test]
    fn test_shell_commands_of_the_shipped_script() {
        let names: Vec<String> = shell_commands(SHELL_INTEGRATION)
            .into_iter()
            .map(|entry| entry.keys)
            .collect();
        assert_eq!(
            names,
            vec!["p".to_string(), "pd".to_string(), "pc".to_string()],
            "These are the functions eval \"$(psychic zsh)\" defines"
        );
    }

    #[test]
    fn test_cli_commands_come_from_clap() {
        let entries = cli_commands();
        let retrain = entries
            .iter()
            .find(|entry| entry.keys == "retrain")
            .expect("retrain should be listed");
        assert!(
            retrain.description.starts_with("retrain the ranking model"),
            "Description comes from the doc comment in cli.rs, got {:?}",
            retrain.description
        );
    }

    #[test]
    fn test_every_row_has_both_columns_filled() {
        for block in blocks() {
            assert!(
                !block.entries.is_empty(),
                "Block {} has no rows",
                block.title
            );
            for entry in block.entries {
                assert!(
                    !entry.keys.is_empty() && !entry.description.is_empty(),
                    "Row {:?} in block {} is missing half of itself",
                    entry,
                    block.title
                );
            }
        }
    }
}

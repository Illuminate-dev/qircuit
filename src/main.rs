use color_eyre::Result;
use ratatui::{
    buffer::Buffer,
    crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    layout::{Constraint, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, Paragraph, Widget},
    DefaultTerminal, Frame,
};

use qircuit::{Gate, QState};

const GATES: &[GateDef] = &[
    GateDef {
        name: "H",
        gate: |q| Gate::H(q),
    },
    GateDef {
        name: "X",
        gate: |q| Gate::X(q),
    },
    GateDef {
        name: "Y",
        gate: |q| Gate::Y(q),
    },
    GateDef {
        name: "Z",
        gate: |q| Gate::Z(q),
    },
    GateDef {
        name: "CNOT",
        gate: |q| Gate::CNOT(vec![q], (q + 1) % N_QUBITS),
    },
    GateDef {
        name: "Measure",
        gate: |_| unreachable!(),
    },
];

const GATE_COLORS: &[Color] = &[
    Color::Cyan,
    Color::Red,
    Color::Magenta,
    Color::Blue,
    Color::Yellow,
    Color::Green,
];

const N_QUBITS: usize = 3;

struct GateDef {
    name: &'static str,
    gate: fn(usize) -> Gate,
}

pub struct App {
    qstate: QState,
    gate_history: Vec<String>,
    selected_gate: usize,
    selected_qubit: usize,
    control_qubits: Vec<usize>,
    status: String,
    exit: bool,
}

impl App {
    fn new() -> Self {
        Self {
            qstate: QState::new(N_QUBITS),
            gate_history: Vec::new(),
            selected_gate: 0,
            selected_qubit: 0,
            control_qubits: Vec::new(),
            status: "ready".to_string(),
            exit: false,
        }
    }

    fn apply_selected_gate(&mut self) {
        let gate_def = &GATES[self.selected_gate];
        let qubit = self.selected_qubit;

        match gate_def.name {
            "Measure" => {
                let result = self.qstate.measure(qubit);
                let msg = format!("Measured qubit {} = {}", qubit, result);
                self.gate_history.push(format!("M({})={}", qubit, result));
                self.status = msg;
            }
            "CNOT" => {
                let controls = self.control_qubits.clone();
                let display = if controls.is_empty() {
                    format!("X({})", qubit)
                } else {
                    format!(
                        "CNOT({}; {})",
                        controls
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(","),
                        qubit
                    )
                };
                self.qstate.apply(&Gate::CNOT(controls, qubit));
                self.gate_history.push(display.clone());
                self.status = display;
            }
            name => {
                let gate = (gate_def.gate)(qubit);
                self.qstate.apply(&gate);
                let display = format!("{}({})", name, qubit);
                self.gate_history.push(display.clone());
                self.status = display;
            }
        }
    }

    fn measure_all(&mut self) {
        let result = self.qstate.measure_all();
        let msg = format!("Measured all = {}", result);
        self.gate_history.push(format!("M_ALL={}", result));
        self.status = msg;
    }

    fn reset(&mut self) {
        self.qstate = QState::new(N_QUBITS);
        self.gate_history.clear();
        self.control_qubits.clear();
        self.status = "reset".to_string();
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('q') => self.exit = true,
            KeyCode::Char('r') => self.reset(),
            KeyCode::Char('c') => {
                if GATES[self.selected_gate].name == "CNOT" {
                    let q = self.selected_qubit;
                    if let Some(pos) = self.control_qubits.iter().position(|&x| x == q) {
                        self.control_qubits.remove(pos);
                    } else {
                        self.control_qubits.push(q);
                        self.control_qubits.sort();
                    }
                }
            }
            KeyCode::Char('M') => self.measure_all(),
            KeyCode::Enter => self.apply_selected_gate(),
            KeyCode::Left => {
                let prev = self.selected_gate;
                self.selected_gate = self.selected_gate.saturating_sub(1);
                if prev != self.selected_gate && GATES[prev].name == "CNOT" {
                    self.control_qubits.clear();
                }
            }
            KeyCode::Right => {
                let prev = self.selected_gate;
                self.selected_gate = (self.selected_gate + 1).min(GATES.len() - 1);
                if prev != self.selected_gate && GATES[prev].name == "CNOT" {
                    self.control_qubits.clear();
                }
            }
            KeyCode::Up => {
                self.selected_qubit = (self.selected_qubit + 1) % N_QUBITS;
            }
            KeyCode::Down => {
                self.selected_qubit = (self.selected_qubit + N_QUBITS - 1) % N_QUBITS;
            }
            _ => {}
        }
    }

    fn handle_events(&mut self) -> Result<()> {
        match event::read()? {
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event);
            }
            _ => {}
        }
        Ok(())
    }

    fn draw(&self, frame: &mut Frame) {
        frame.render_widget(self, frame.area());
    }

    fn run(mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.exit {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events()?;
        }
        Ok(())
    }
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::bordered()
            .title(" Qircuit ".bold())
            .title(Line::from(format!(" {} qubits ", N_QUBITS)).right_aligned());
        let inner = block.inner(area);
        block.render(area, buf);

        let [state_area, controls_area, footer_area] = Layout::vertical([
            Constraint::Fill(1),
            Constraint::Length(5),
            Constraint::Length(1),
        ])
        .areas(inner);

        self.render_state(state_area, buf);
        self.render_controls(controls_area, buf);
        self.render_footer(footer_area, buf);
    }
}

impl App {
    fn render_state(&self, area: Rect, buf: &mut Buffer) {
        let n = self.qstate.n;
        let state = &self.qstate.state;
        let bar_width = (area.width as usize).saturating_sub(28).max(4);

        let lines: Vec<Line> = (0..state.len())
            .map(|i| {
                let amp = state[i];
                let prob = amp.norm_sqr();
                let filled = (prob * bar_width as f64).round() as usize;
                let bar_filled = "█".repeat(filled.min(bar_width));
                let bar_empty = "░".repeat(bar_width.saturating_sub(filled));

                Line::from(vec![
                    Span::styled(format!(" |{:0n$b}⟩ ", i), Color::Cyan),
                    Span::styled(bar_filled, Color::Green),
                    Span::styled(bar_empty, Color::DarkGray),
                    Span::raw(format!(" {:>6.3}  ", prob)),
                    Span::styled(format!("{:>6.3}{:>+.3}i", amp.re, amp.im), Color::Yellow),
                ])
            })
            .collect();

        Paragraph::new(Text::from(lines)).render(area, buf);
    }

    fn render_controls(&self, area: Rect, buf: &mut Buffer) {
        let gate_line = Line::from({
            let mut spans = vec![Span::styled(" Gate: ", Color::White)];
            for (i, g) in GATES.iter().enumerate() {
                let color = GATE_COLORS[i];
                let text = if i == self.selected_gate {
                    format!(" [{}] ", g.name)
                } else {
                    format!("  {}  ", g.name)
                };
                spans.push(if i == self.selected_gate {
                    Span::styled(text, Style::new().fg(color).bold())
                } else {
                    Span::styled(text, color)
                });
            }
            spans
        });

        let qubit_line = Line::from({
            let mut spans = vec![Span::styled(" Qubit:", Color::White)];
            let cnot_active = GATES[self.selected_gate].name == "CNOT";
            if cnot_active {
                for i in 0..N_QUBITS {
                    let is_control = self.control_qubits.contains(&i);
                    let is_cursor = i == self.selected_qubit;
                    let label = if is_control {
                        format!(" [{}]ctl", i)
                    } else if is_cursor {
                        format!(" [{}]tgt", i)
                    } else {
                        format!("  {}  ", i)
                    };
                    let color = if is_control || is_cursor {
                        Color::Yellow
                    } else {
                        Color::White
                    };
                    spans.push(Span::styled(label, color));
                }
            } else {
                for i in 0..N_QUBITS {
                    let text = if i == self.selected_qubit {
                        format!(" [{}] ", i)
                    } else {
                        format!("  {}  ", i)
                    };
                    let color = if i == self.selected_qubit {
                        Color::Yellow
                    } else {
                        Color::White
                    };
                    spans.push(Span::styled(text, color));
                }
            }
            spans
        });

        let max_items = 6;
        let history_text = if self.gate_history.is_empty() {
            " (no gates applied)".to_string()
        } else if self.gate_history.len() > max_items {
            let start = self.gate_history.len() - max_items;
            format!(" …{}", self.gate_history[start..].join(" → "))
        } else {
            self.gate_history.join(" → ")
        };
        let history_line = Line::from(vec![
            Span::styled(" History:", Color::White),
            Span::raw(format!(" {}", history_text)),
        ]);

        let status_line = Line::from(vec![
            Span::styled(" Status:", Color::White),
            Span::styled(format!(" {}", self.status), Color::Green),
        ]);

        let controls = Text::from(vec![
            Line::from(""),
            gate_line,
            qubit_line,
            history_line,
            status_line,
        ]);
        Paragraph::new(controls).render(area, buf);
    }

    fn render_footer(&self, area: Rect, buf: &mut Buffer) {
        let text = Line::from(vec![
            " q:quit ".into(),
            " r:reset ".into(),
            " Enter:apply ".into(),
            " ←→:gate ".into(),
            " ↑↓:qubit ".into(),
            " c:ctl_toggle ".into(),
            " M:measure_all ".into(),
        ]);
        Paragraph::new(Text::from(text)).render(area, buf);
    }
}

pub fn main() -> Result<()> {
    color_eyre::install()?;
    ratatui::run(|term| App::new().run(term))?;
    Ok(())
}

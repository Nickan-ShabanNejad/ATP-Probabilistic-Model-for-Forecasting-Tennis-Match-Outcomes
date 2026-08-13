DIFF_FEATURES = [
    "overall_elo_diff", "surface_elo_diff", "serve_diff", "return_diff",
    "log_rank_advantage", "win3_diff", "win5_diff", "win10_diff", "surface_win10_diff",
    "spw1_diff", "spw3_diff", "spw5_diff", "spw10_diff",
    "rpw1_diff", "rpw3_diff", "rpw5_diff", "rpw10_diff",
    "first_in5_diff", "first_won5_diff", "second_won5_diff", "ace_rate5_diff", "df_rate5_diff",
    "point_share5_diff", "point_share10_diff", "bp_save5_diff", "bp_convert5_diff",
    "form_ewma_diff", "surface_form_ewma_diff", "opp_elo10_diff", "recent_perf10_diff",
    "matches7_diff", "matches14_diff", "rest_days_diff", "elo_change10_diff", "age_diff",
    "winner_rate_diff", "ue_rate_diff", "aggression_quality_diff", "advanced_coverage_diff",
    "net_win_diff", "avg_first_serve_speed_diff",
    "h2h_overall_edge", "h2h_surface_edge", "h2h_serve_diff", "h2h_second_serve_diff", "h2h_bp_convert_diff",
    "level_surface_elo_interaction", "level_rank_interaction", "level_serve_interaction", "level_form_interaction",
    "speed_surface_elo_interaction", "speed_serve_interaction", "speed_return_interaction",
    "speed_ace_interaction", "speed_second_serve_interaction", "speed_point_share_interaction",
    "indoor_serve_interaction", "indoor_return_interaction", "bestof_surface_elo_interaction",
]

CONTEXT_FEATURES = [
    "tournament_level", "best_of", "court_speed", "court_speed_prior", "court_speed_live_weight",
    "court_speed_missing", "indoor", "h2h_matches_log", "h2h_surface_matches_log",
]

FEATURES = DIFF_FEATURES + CONTEXT_FEATURES

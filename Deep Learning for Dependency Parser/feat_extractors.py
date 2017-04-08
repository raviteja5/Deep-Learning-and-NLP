class SimpleFeatureExtractor:

    def get_features(self, parser_state, **kwargs):
        """
        The features used as input to the action chooser network to 
        decide the next action.
        """
        #if parser_state.input_buffer_len() == 0:
        #    print parser_state.sentence
        st_entries = parser_state.stack_peek_n(2)
        st_entries.append(parser_state.input_buffer_peek_n(1)[0])
        return [s.embedding for s in st_entries]
